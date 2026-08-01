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
parser = argparse.ArgumentParser(description="Phase 6: Causal Baseline (No KD)")
parser.add_argument("--data_dir", type=str, default="./data", help="Path to CIFAR-10")
parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
parser.add_argument("--drw_epoch", type=int, default=80, help="Epoch to start Deferred Reweighting")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size (Matches Phase 1)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Matches Phase 1)")
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
    print(f"Detected {torch.cuda.device_count()} GPUs! Enabled CuDNN Benchmark.")

# ==========================================
# 2. Imbalanced CIFAR-10 Construction
# ==========================================
def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = 5000 
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

print(f"Preparing Imbalanced CIFAR-10-LT (IF={args.imb_factor})...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224), # Resize for ViT
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224), # Resize for ViT
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

img_num_per_cls = get_img_num_per_cls(10, 'exp', args.imb_factor)
train_indices = gen_imbalanced_data(train_dataset.targets, img_num_per_cls)

train_dataset.data = train_dataset.data[train_indices]
train_dataset.targets = np.array(train_dataset.targets)[train_indices].tolist()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

class_counts = img_num_per_cls
head_classes = [i for i in range(10) if class_counts[i] > 1500]
med_classes = [i for i in range(10) if 100 < class_counts[i] <= 1500]
tail_classes = [i for i in range(10) if class_counts[i] <= 100]

# DRW Weights
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * 10
per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

# ==========================================
# 3. Model Definition (No Teacher)
# ==========================================
class DeiTLT(nn.Module):
    def __init__(self, model_type, num_classes=10):
        super().__init__()
        # Load the architecture (DeiT natively has CLS and DIST tokens if we use a distilled variant)
        # We must use the distilled variant architecture to ensure the DIST token exists.
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
    targets_a_oh = F.one_hot(targets, num_classes=10).float()
    targets_b_oh = F.one_hot(targets[index], num_classes=10).float()
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
# 4. Training (Without KD)
# ==========================================
arch = 'deit_tiny_patch16_224'
print("\n" + "="*60)
print(f"STARTING CAUSAL EXPERIMENT: NO KNOWLEDGE DISTILLATION")
print(f"Model: {arch}")
print("="*60)

model = DeiTLT(arch, num_classes=10).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Notice: NO TEACHER is loaded!

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler('cuda')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

all_test_lbls = []
all_test_logits_cls = []
all_test_logits_dist = []

for epoch in range(args.epochs):
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
            targets = F.one_hot(targets, num_classes=10).float()
            
        with torch.amp.autocast('cuda'):
            logits_cls, logits_dist = model(images)
            
            # Since there is NO TEACHER, the DIST token is forced to learn directly from Ground Truth
            # exactly the same way the CLS token does.
            if use_drw:
                loss_cls = F.cross_entropy(logits_cls, targets.argmax(dim=1), weight=per_cls_weights)
                loss_dist = F.cross_entropy(logits_dist, targets.argmax(dim=1), weight=per_cls_weights)
            else:
                loss_cls = base_criterion(logits_cls, targets)
                loss_dist = base_criterion(logits_dist, targets)
                
            loss = loss_cls + loss_dist
            
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
        head_correct_cls, tail_correct_cls = 0, 0
        head_correct_dist, tail_correct_dist = 0, 0
        head_total, tail_total = 0, 0
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
                
                for c_idx in range(10):
                    mask = (lbls == c_idx)
                    n_c = mask.sum().item()
                    c_corr_cls = (p_cls[mask] == c_idx).sum().item()
                    c_corr_dist = (p_dist[mask] == c_idx).sum().item()
                    
                    if c_idx in head_classes: 
                        head_total += n_c
                        head_correct_cls += c_corr_cls
                        head_correct_dist += c_corr_dist
                    elif c_idx in tail_classes: 
                        tail_total += n_c
                        tail_correct_cls += c_corr_cls
                        tail_correct_dist += c_corr_dist
                total += lbls.size(0)
                
        cls_acc = cls_correct / total * 100
        dist_acc = dist_correct / total * 100
        avg_acc = avg_correct / total * 100
        
        print(f"Ep {epoch+1:03d} [{epoch_time:.1f}s] | Acc:[CLS:{cls_acc:.1f} DIST:{dist_acc:.1f}]")
        print(f"           | CLS Tail: {(tail_correct_cls/tail_total*100) if tail_total>0 else 0:.1f}%")
        print(f"           | DIST Tail: {(tail_correct_dist/tail_total*100) if tail_total>0 else 0:.1f}%")

# Train Neural Router
print(f"\nTraining Neural Router...")
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

print("\n" + "="*50)
print("PHASE 6 CAUSAL EXPERIMENT (NO-KD) RESULTS")
print("="*50)
print(f"CLS Token Native Acc:  {cls_acc:.2f}%")
print(f"DIST Token Native Acc: {dist_acc:.2f}%")
print(f"50/50 Baseline Acc:    {avg_acc:.2f}%")
print(f"Neural Router Acc:     {router_acc:.2f}%")
print("\nROUTER ANALYSIS:")
avg_alpha = alphas.mean().item()
print(f"Average Route Alpha: {avg_alpha:.3f} (If this is ~0.5, then KD was the cause!)")
print("="*50)
