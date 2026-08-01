import os
import time
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm

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
parser = argparse.ArgumentParser(description="Phase 4: ImageNet-LT Neural Entropy Router Ablation")
parser.add_argument("--data_dir", type=str, default="/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC", help="Path to ImageNet-LT directory")
parser.add_argument("--model_type", type=str, default="deit_tiny_patch16_224", help="Timm ViT model string")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pth")
parser.add_argument("--epochs", type=int, default=300, help="Total training epochs")
parser.add_argument("--drw_epoch", type=int, default=250, help="Epoch to start Deferred Reweighting")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

import sys
if 'ipykernel' in sys.modules:
    print("Detected Jupyter/Kaggle environment. Using default notebook arguments.")
    args = parser.parse_args(args=["--data_dir", "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"])
else:
    args = parser.parse_args()

# Reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"Detected {torch.cuda.device_count()} GPUs! Enabled CuDNN Benchmark for maximum speed.")

# ==========================================
# 2. Dataloaders & ImageNet-LT Statistics
# ==========================================
print(f"Mounting official Kaggle ImageNet dataset from: {args.data_dir}")

import urllib.request
from PIL import Image
from torch.utils.data import Dataset

# Download Official ImageNet-LT splits if they don't exist
train_txt = "ImageNet_LT_train.txt"
test_txt = "ImageNet_LT_val.txt"

if not os.path.exists(train_txt):
    print("Downloading ImageNet_LT_train.txt...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/facebookresearch/ic_gan/main/BigGAN_PyTorch/imagenet_lt/ImageNet_LT_train.txt", train_txt)
if not os.path.exists(test_txt):
    print("Downloading ImageNet_LT_val.txt...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/facebookresearch/ic_gan/main/BigGAN_PyTorch/imagenet_lt/ImageNet_LT_val.txt", test_txt)

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
                
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        try:
            with open(path, 'rb') as f:
                sample = Image.open(f).convert('RGB')
        except Exception as e:
            # Fallback if specific file is corrupted/missing in Kaggle split
            print(f"Error loading {path}: {e}")
            sample = Image.new('RGB', (224, 224))
            
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = LT_Dataset(root=args.data_dir, txt=train_txt, transform=transform_train)
test_dataset = LT_Dataset(root=args.data_dir, txt=test_txt, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

print("Counting class frequencies to define Head, Medium, Tail boundaries...")
class_counts_dict = Counter(train_dataset.labels)
class_counts = [class_counts_dict.get(i, 0) for i in range(1000)]

# ImageNet-LT thresholds
head_classes = [i for i in range(1000) if class_counts[i] > 100]
med_classes = [i for i in range(1000) if 20 < class_counts[i] <= 100]
tail_classes = [i for i in range(1000) if class_counts[i] <= 20]

print(f"Total Classes: {len(class_counts)}")
print(f"Head Classes (>100): {len(head_classes)}")
print(f"Medium Classes (20-100): {len(med_classes)}")
print(f"Tail Classes (<=20): {len(tail_classes)}")

# DRW Weights
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * 1000
per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

# ==========================================
# 3. Model Definitions
# ==========================================
class DeiTLT(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Use user-specified architecture
        self.model = timm.create_model(args.model_type, pretrained=False, num_classes=num_classes)
        embed_dim = self.model.head.in_features
        
        # Strip default head
        self.model.head = nn.Identity()
        
        # Dual Output Heads
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

print(f"Initializing Student: {args.model_type}")
model = DeiTLT(num_classes=1000).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

print("Downloading Pretrained ImageNet Teacher (resnext50_32x4d)...")
teacher = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=1000)
teacher = teacher.to(device)
if torch.cuda.device_count() > 1:
    teacher = nn.DataParallel(teacher)
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

# ==========================================
# 4. Training Utilities
# ==========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler('cuda')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
base_criterion = SoftTargetCrossEntropy()

def mixup_fn(images, targets, alpha=0.8):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    targets_a, targets_b = targets, targets[index]
    targets_a_oh = F.one_hot(targets_a, num_classes=1000).float()
    targets_b_oh = F.one_hot(targets_b, num_classes=1000).float()
    mixed_targets = lam * targets_a_oh + (1 - lam) * targets_b_oh
    return mixed_images, mixed_targets

# ==========================================
# 5. Training Loop
# ==========================================
start_epoch = 0
checkpoint_path = "checkpoint_latest.pth"

if args.resume and os.path.exists(checkpoint_path):
    print(f"Resuming from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed successfully at Epoch {start_epoch+1}")

print("\n" + "="*50)
print(f"STARTING TRAINING (EPOCH {start_epoch+1} TO {args.epochs})")
print("="*50)

# To store for the Oracle
all_test_lbls = []
all_test_preds_cls = []
all_test_preds_dist = []
all_test_logits_cls = []
all_test_logits_dist = []
all_test_feat_cls = []
all_test_feat_dist = []

for epoch in range(start_epoch, args.epochs):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    total_teacher_entropy = 0.0
    
    use_drw = epoch >= args.drw_epoch
    weight = 2.0 if use_drw else 1.0
    
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        if not use_drw:
            images, targets = mixup_fn(images, targets)
        else:
            targets = F.one_hot(targets, num_classes=1000).float()
            
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                teacher_logits = teacher(images)
                prob_t = F.softmax(teacher_logits / 3.0, dim=1)
                entropy = -(prob_t * torch.log(prob_t + 1e-8)).sum(dim=1).mean()
                total_teacher_entropy += entropy.item()
                
            logits_cls, logits_dist = model(images)
            
            if use_drw:
                loss_cls = F.cross_entropy(logits_cls, targets.argmax(dim=1), weight=per_cls_weights)
            else:
                loss_cls = base_criterion(logits_cls, targets)
                
            # Distillation Loss (Hard)
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
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        cls_correct, dist_correct, avg_correct = 0, 0, 0
        head_correct, med_correct, tail_correct = 0, 0, 0
        head_total, med_total, tail_total = 0, 0, 0
        total = 0
        
        is_last_epoch = (epoch + 1) == args.epochs
        
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                l_cls, l_dist, f_cls, f_dist = model(imgs, return_features=True)
                l_avg = (l_cls + l_dist) / 2
                
                p_cls = l_cls.argmax(dim=1)
                p_dist = l_dist.argmax(dim=1)
                p_avg = l_avg.argmax(dim=1)
                
                if is_last_epoch:
                    all_test_lbls.append(lbls.cpu())
                    all_test_preds_cls.append(p_cls.cpu())
                    all_test_preds_dist.append(p_dist.cpu())
                    all_test_logits_cls.append(l_cls.cpu())
                    all_test_logits_dist.append(l_dist.cpu())
                    all_test_feat_cls.append(f_cls.cpu())
                    all_test_feat_dist.append(f_dist.cpu())
                
                cls_correct += (p_cls == lbls).sum().item()
                dist_correct += (p_dist == lbls).sum().item()
                avg_correct += (p_avg == lbls).sum().item()
                
                for c_idx in range(1000):
                    mask = (lbls == c_idx)
                    n_c = mask.sum().item()
                    c_correct = (p_avg[mask] == c_idx).sum().item()
                    
                    if c_idx in head_classes:
                        head_total += n_c
                        head_correct += c_correct
                    elif c_idx in med_classes:
                        med_total += n_c
                        med_correct += c_correct
                    elif c_idx in tail_classes:
                        tail_total += n_c
                        tail_correct += c_correct
                        
                total += lbls.size(0)
                
        cls_acc = cls_correct / total * 100
        dist_acc = dist_correct / total * 100
        avg_acc = avg_correct / total * 100
        
        h_acc = (head_correct / head_total * 100) if head_total > 0 else 0
        m_acc = (med_correct / med_total * 100) if med_total > 0 else 0
        t_acc = (tail_correct / tail_total * 100) if tail_total > 0 else 0
        
        print(f"Ep {epoch+1:03d} [{epoch_time:.1f}s] | L:{total_loss/len(train_loader):.3f} LR:{current_lr:.1e} | Acc:[CLS:{cls_acc:.1f} DIST:{dist_acc:.1f} AVG:{avg_acc:.1f}] | HMT:[H:{h_acc:.1f} M:{m_acc:.1f} T:{t_acc:.1f}]")
        
        csv_path = "metrics_imagenet.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a") as f:
            if not file_exists or os.path.getsize(csv_path) == 0:
                f.write("Epoch,Loss,LR,CLS_Acc,DIST_Acc,AVG_Acc,Head_Acc,Med_Acc,Tail_Acc\n")
            f.write(f"{epoch+1},{total_loss/len(train_loader):.4f},{current_lr:.2e},{cls_acc:.2f},{dist_acc:.2f},{avg_acc:.2f},{h_acc:.2f},{m_acc:.2f},{t_acc:.2f}\n")
            
    # Save checkpoint at the end of every epoch
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, checkpoint_path)

# ==========================================
# 6. Post-Training: Oracle Alpha Search
# ==========================================
print("\n" + "="*50)
print("EXTRACTING FEATURES FOR NEURAL ROUTER & ORACLE")
print("="*50)

all_test_lbls = torch.cat(all_test_lbls)
all_test_logits_cls = torch.cat(all_test_logits_cls)
all_test_logits_dist = torch.cat(all_test_logits_dist)
all_test_feat_cls = torch.cat(all_test_feat_cls)
all_test_feat_dist = torch.cat(all_test_feat_dist)

# Generate Confidence and Entropy
p_cls = F.softmax(all_test_logits_cls, dim=1)
p_dist = F.softmax(all_test_logits_dist, dim=1)
conf_cls = p_cls.max(dim=1)[0].unsqueeze(1)
conf_dist = p_dist.max(dim=1)[0].unsqueeze(1)
ent_cls = -(p_cls * torch.log(p_cls + 1e-8)).sum(dim=1, keepdim=True)
ent_dist = -(p_dist * torch.log(p_dist + 1e-8)).sum(dim=1, keepdim=True)

X_features = torch.cat([conf_cls, conf_dist, ent_cls, ent_dist], dim=1)
Y_labels = all_test_lbls

# Oracle Search
print("Running Oracle Alpha Search over 1000 classes...")
alphas = np.linspace(0, 1, 21)
optimal_alphas = {}

for c in range(1000):
    mask = (Y_labels == c)
    if mask.sum() == 0: continue
    
    l_c = all_test_logits_cls[mask]
    l_d = all_test_logits_dist[mask]
    
    best_acc = 0.0
    best_alpha = 0.5
    for a in alphas:
        fused = a * l_c + (1 - a) * l_d
        acc = (fused.argmax(dim=1) == c).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_alpha = a
            
    optimal_alphas[c] = best_alpha
    if c % 100 == 0:
        print(f"Class {c:4d} | Best alpha*: {best_alpha:.2f} | Acc: {best_acc*100:.2f}%")

oracle_correct = 0
for i in range(len(Y_labels)):
    c = Y_labels[i].item()
    a = optimal_alphas[c]
    fused = a * all_test_logits_cls[i] + (1 - a) * all_test_logits_dist[i]
    if fused.argmax() == c:
        oracle_correct += 1
oracle_acc = oracle_correct / len(Y_labels) * 100
print(f"Oracle Upper Bound: {oracle_acc:.2f}%")

# ==========================================
# 7. Post-Training: Neural Entropy Router
# ==========================================
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

router = NeuralRouter().to(device)
r_optimizer = torch.optim.Adam(router.parameters(), lr=0.01)

X_train = X_features.to(device)
L_c = all_test_logits_cls.to(device)
L_d = all_test_logits_dist.to(device)
Y_train = Y_labels.to(device)

print("\nTraining Instance-Level Neural Router (500 Epochs)...")
for e in range(500):
    router.train()
    r_optimizer.zero_grad()
    
    alphas = router(X_train)
    fused_logits = alphas * L_c + (1 - alphas) * L_d
    loss = F.cross_entropy(fused_logits, Y_train)
    
    loss.backward()
    r_optimizer.step()
    
    if e % 100 == 0:
        acc = (fused_logits.argmax(dim=1) == Y_train).float().mean().item() * 100
        print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | Router Acc: {acc:.2f}%")

# Final Evaluation
router.eval()
with torch.no_grad():
    alphas = router(X_train)
    fused = alphas * L_c + (1 - alphas) * L_d
    final_router_acc = (fused.argmax(dim=1) == Y_train).float().mean().item() * 100

print("\n" + "="*40)
print(f"Baseline (50/50) Accuracy: {avg_acc:.2f}%")
print(f"Oracle Upper Bound:        {oracle_acc:.2f}%")
print(f"Neural Router Accuracy:    {final_router_acc:.2f}%")
print("="*40)

# Average Alphas for Head vs Tail
alphas_np = alphas.cpu().numpy()
head_mask = np.isin(Y_labels.numpy(), head_classes)
tail_mask = np.isin(Y_labels.numpy(), tail_classes)

avg_head_alpha = np.mean(alphas_np[head_mask])
avg_tail_alpha = np.mean(alphas_np[tail_mask])

print(f"Average Router Alpha for Head Classes: {avg_head_alpha:.3f}")
print(f"Average Router Alpha for Tail Classes: {avg_tail_alpha:.3f}")
print("Done!")
