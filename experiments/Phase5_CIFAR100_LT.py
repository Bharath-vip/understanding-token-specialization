import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm

# ==========================================
# 1. Argument Parser
# ==========================================
parser = argparse.ArgumentParser(description="Phase 5: CIFAR-100-LT Architecture Scale Ablation")
parser.add_argument("--data_dir", type=str, default="./data", help="Path to download CIFAR-100")
parser.add_argument("--model", type=str, default="deit_tiny_patch16_224", help="Architecture to run")
parser.add_argument("--epochs", type=int, default=100, help="Total training epochs per model")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_phase5.pth")
parser.add_argument("--drw_epoch", type=int, default=80, help="Epoch to start Deferred Reweighting")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (High for 2x T4 GPUs)")
parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--imb_factor", type=int, default=50, help="Imbalance factor (Max/Min)")

import sys
if 'ipykernel' in sys.modules:
    print("Detected Jupyter/Kaggle environment. Using default notebook arguments.")
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

# Reproducibility & Performance
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"Detected {torch.cuda.device_count()} GPUs! Enabled CuDNN Benchmark for maximum speed.")

# ==========================================
# 2. Imbalanced CIFAR-100 Construction
# ==========================================
def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = 500 # CIFAR-100 has 500 images per class in train set
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (1.0 / imb_factor) ** (cls_idx / (cls_num - 1.0))
            img_num_per_cls.append(int(num))
    return img_num_per_cls

def gen_imbalanced_data(targets, img_num_per_cls):
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)
    num_per_cls_dict = dict(zip(classes, img_num_per_cls))
    
    new_indices = []
    for the_class, the_img_num in num_per_cls_dict.items():
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_indices.extend(selec_idx)
    return new_indices

print("Preparing Imbalanced CIFAR-100-LT Dataset...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224), # Resize for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

transform_test = transforms.Compose([
    transforms.Resize(224), # Resize for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)

img_num_per_cls = get_img_num_per_cls(100, 'exp', args.imb_factor)
train_indices = gen_imbalanced_data(train_dataset.targets, img_num_per_cls)

train_dataset.data = train_dataset.data[train_indices]
train_dataset.targets = np.array(train_dataset.targets)[train_indices].tolist()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

class_counts = img_num_per_cls
head_classes = [i for i in range(100) if class_counts[i] > 100]
med_classes = [i for i in range(100) if 20 < class_counts[i] <= 100]
tail_classes = [i for i in range(100) if class_counts[i] <= 20]

print(f"Total Training Images: {len(train_dataset)}")
print(f"Total Classes: {len(class_counts)}")
print(f"Head Classes (>100 imgs): {len(head_classes)}")
print(f"Medium Classes (20-100 imgs): {len(med_classes)}")
print(f"Tail Classes (<=20 imgs): {len(tail_classes)}")

# DRW Weights
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * 100
per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

# ==========================================
# 3. Training Utilities
# ==========================================
class DeiTLT(nn.Module):
    def __init__(self, model_type, num_classes=100):
        super().__init__()
        self.model = timm.create_model(model_type, pretrained=True, num_classes=num_classes)
        embed_dim = self.model.head.in_features
        self.model.head = nn.Identity()
        self.head_cls = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, return_features=False):
        features = self.model.forward_features(x)
        cls_token = features[:, 0]
        dist_token = features[:, 1]
        logits_cls = self.head_cls(cls_token)
        logits_dist = self.head_dist(dist_token)
        if return_features:
            return logits_cls, logits_dist, cls_token, dist_token
        return logits_cls, logits_dist

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
base_criterion = SoftTargetCrossEntropy()

def mixup_fn(images, targets, alpha=0.8):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    targets_a_oh = F.one_hot(targets, num_classes=100).float()
    targets_b_oh = F.one_hot(targets[index], num_classes=100).float()
    mixed_targets = lam * targets_a_oh + (1 - lam) * targets_b_oh
    return mixed_images, mixed_targets

class NeuralRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(x)

# ==========================================
# 4. Master Ablation Execution
# ==========================================
arch = args.model
results = []

print("\n" + "="*60)
print(f"STARTING ABLATION: {arch}")
print("="*60)

# Init Models
model = DeiTLT(arch, num_classes=100).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
teacher_name = 'resnext50_32x4d' if arch != 'deit_base_patch16_224' else 'resnet101'
print(f"Loading Teacher: {teacher_name}")
teacher = timm.create_model(teacher_name, pretrained=True, num_classes=1000)
teacher.fc = nn.Linear(teacher.fc.in_features, 100) # Re-head for CIFAR-100
teacher = teacher.to(device)
if torch.cuda.device_count() > 1:
    teacher = nn.DataParallel(teacher)
teacher.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler('cuda')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training Loop
all_test_lbls = []
all_test_logits_cls = []
all_test_logits_dist = []

start_epoch = 0
checkpoint_path = f"checkpoint_phase5_{arch}.pth"
if args.resume and os.path.exists(checkpoint_path):
    print(f"Resuming {arch} from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed successfully at Epoch {start_epoch+1}")
    
for epoch in range(start_epoch, args.epochs):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    use_drw = epoch >= args.drw_epoch
    weight = 2.0 if use_drw else 1.0
    
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        if not use_drw:
            images, targets = mixup_fn(images, targets)
        else:
            targets = F.one_hot(targets, num_classes=100).float()
            
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            logits_cls, logits_dist = model(images)
            
            if use_drw:
                loss_cls = F.cross_entropy(logits_cls, targets.argmax(dim=1), weight=per_cls_weights)
            else:
                loss_cls = base_criterion(logits_cls, targets)
                
            t_targets = teacher_logits.argmax(dim=1)
            loss_dist = F.cross_entropy(logits_dist, t_targets)
            loss = loss_cls + (weight * loss_dist)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    epoch_time = time.time() - epoch_start_time
    
    if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
        model.eval()
        cls_correct, dist_correct, avg_correct = 0, 0, 0
        head_correct, med_correct, tail_correct = 0, 0, 0
        head_total, med_total, tail_total = 0, 0, 0
        total = 0
        is_last_epoch = (epoch + 1) == args.epochs
        
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                l_cls, l_dist, _, _ = model(imgs, return_features=True)
                l_avg = (l_cls + l_dist) / 2
                
                p_cls, p_dist, p_avg = l_cls.argmax(dim=1), l_dist.argmax(dim=1), l_avg.argmax(dim=1)
                
                if is_last_epoch:
                    all_test_lbls.append(lbls.cpu())
                    all_test_logits_cls.append(l_cls.cpu())
                    all_test_logits_dist.append(l_dist.cpu())
                
                cls_correct += (p_cls == lbls).sum().item()
                dist_correct += (p_dist == lbls).sum().item()
                avg_correct += (p_avg == lbls).sum().item()
                
                for c_idx in range(100):
                    mask = (lbls == c_idx)
                    n_c = mask.sum().item()
                    c_correct = (p_avg[mask] == c_idx).sum().item()
                    
                    if c_idx in head_classes: head_total += n_c; head_correct += c_correct
                    elif c_idx in med_classes: med_total += n_c; med_correct += c_correct
                    elif c_idx in tail_classes: tail_total += n_c; tail_correct += c_correct
                total += lbls.size(0)
                
        cls_acc = cls_correct / total * 100
        dist_acc = dist_correct / total * 100
        avg_acc = avg_correct / total * 100
        h_acc = (head_correct / head_total * 100) if head_total > 0 else 0
        m_acc = (med_correct / med_total * 100) if med_total > 0 else 0
        t_acc = (tail_correct / tail_total * 100) if tail_total > 0 else 0
        
        print(f"Ep {epoch+1:03d} [{epoch_time:.1f}s] | Acc:[CLS:{cls_acc:.1f} DIST:{dist_acc:.1f} AVG:{avg_acc:.1f}] | HMT:[H:{h_acc:.1f} M:{m_acc:.1f} T:{t_acc:.1f}]")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, checkpoint_path)

# Train Neural Router
print(f"\nTraining Neural Router for {arch}...")
Y_labels = torch.cat(all_test_lbls)
L_c = torch.cat(all_test_logits_cls)
L_d = torch.cat(all_test_logits_dist)

p_cls, p_dist = F.softmax(L_c, dim=1), F.softmax(L_d, dim=1)
conf_cls, conf_dist = p_cls.max(dim=1)[0].unsqueeze(1), p_dist.max(dim=1)[0].unsqueeze(1)
ent_cls = -(p_cls * torch.log(p_cls + 1e-8)).sum(dim=1, keepdim=True)
ent_dist = -(p_dist * torch.log(p_dist + 1e-8)).sum(dim=1, keepdim=True)

X_features = torch.cat([conf_cls, conf_dist, ent_cls, ent_dist], dim=1)

router = NeuralRouter().to(device)
r_optimizer = torch.optim.Adam(router.parameters(), lr=0.01)
X_train, Y_train = X_features.to(device), Y_labels.to(device)
L_c_d, L_d_d = L_c.to(device), L_d.to(device)

for e in range(500):
    router.train()
    r_optimizer.zero_grad()
    alphas = router(X_train)
    fused = alphas * L_c_d + (1 - alphas) * L_d_d
    loss = F.cross_entropy(fused, Y_train)
    loss.backward()
    r_optimizer.step()
    
router.eval()
with torch.no_grad():
    alphas = router(X_train)
    fused = alphas * L_c_d + (1 - alphas) * L_d_d
    router_acc = (fused.argmax(dim=1) == Y_train).float().mean().item() * 100
    
# Oracle Search
alphas_grid = np.linspace(0, 1, 21)
oracle_correct = 0
for c in range(100):
    mask = (Y_labels == c)
    if mask.sum() == 0: continue
    best_acc, best_alpha = 0, 0.5
    l_c_mask, l_d_mask = L_c[mask], L_d[mask]
    for a in alphas_grid:
        f = a * l_c_mask + (1 - a) * l_d_mask
        acc = (f.argmax(dim=1) == c).float().mean().item()
        if acc > best_acc: best_acc, best_alpha = acc, a
    for i in np.where(mask)[0]:
        f = best_alpha * L_c[i] + (1 - best_alpha) * L_d[i]
        if f.argmax() == c: oracle_correct += 1
oracle_acc = oracle_correct / len(Y_labels) * 100

print(f"\n[RESULT {arch}]")
print(f"50/50 Baseline: {avg_acc:.2f}%")
print(f"Oracle Bound:   {oracle_acc:.2f}%")
print(f"Neural Router:  {router_acc:.2f}%")
print("-"*40)

results.append({
    'Model': arch,
    'CLS_Acc': cls_acc,
    'DIST_Acc': dist_acc,
    'Baseline_5050': avg_acc,
    'Router_Acc': router_acc,
    'Oracle_Bound': oracle_acc
})

print("\n" + "="*50)
print("PHASE 5 CIFAR-100-LT FINAL RESULTS")
print("="*50)
df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv('Phase5_CIFAR100_LT_Results.csv', index=False)
