"""
Learning rate sweep v2 for EfficientNetV2-L - exploring around best configs.

Best from v1:
- aggressive_unfreeze (75.4%): lr_frozen=1e-3, lr_unfrozen=5e-5→5e-4, 12ep
- high_everywhere (75.0%): lr_frozen=2e-3, lr_unfrozen=1e-4→1e-3, 12ep

This sweep explores combinations and variations around these winners.

Usage:
    modal run sweep_lr_xu2014_v2.py
    modal run sweep_lr_xu2014_v2.py --tiny
"""

import modal
from pathlib import Path
import random
import numpy as np

# Modal setup
APP_NAME = "archstyle-lr-sweep-v2"
DATA_VOL_NAME = "archstyle-data-xu2014"
RESULTS_VOL_NAME = "archstyle-xu2014-results-v2"

REMOTE_DATA_DIR = Path("/data")
REMOTE_OUTPUT_DIR = Path("/output")

app = modal.App(name=APP_NAME)

data_volume = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy",
        "pillow",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    )
)

# 8 LR configurations with descriptive names
# Format: (name, lr_frozen, lr_unfrozen_min, lr_unfrozen_max, epochs)
SWEEP_CONFIGS = [
    # Baseline (for comparison)
    ("frozen1e3_unfrozen1e5to1e4_12ep", 1e-3, 1e-5, 1e-4, 12),
    
    # Variations around aggressive_unfreeze (winner at 75.4%)
    # Original: lr_frozen=1e-3, lr_unfrozen=5e-5→5e-4
    
    # Try higher frozen LR with aggressive unfrozen
    ("frozen2e3_unfrozen5e5to5e4_12ep", 2e-3, 5e-5, 5e-4, 12),
    
    # Try even higher unfrozen (between aggressive and high_everywhere)
    ("frozen1e3_unfrozen1e4to5e4_12ep", 1e-3, 1e-4, 5e-4, 12),
    
    # Aggressive unfrozen with longer training
    ("frozen1e3_unfrozen5e5to5e4_18ep", 1e-3, 5e-5, 5e-4, 18),
    
    # Variations around high_everywhere (75.0%)
    # Original: lr_frozen=2e-3, lr_unfrozen=1e-4→1e-3
    
    # High everywhere with slightly lower unfrozen max
    ("frozen2e3_unfrozen1e4to5e4_12ep", 2e-3, 1e-4, 5e-4, 12),
    
    # High everywhere with mid frozen
    ("frozen15e3_unfrozen1e4to1e3_12ep", 1.5e-3, 1e-4, 1e-3, 12),
    
    # Combinations / new explorations
    
    # Mid-aggressive: between baseline and aggressive
    ("frozen1e3_unfrozen3e5to3e4_12ep", 1e-3, 3e-5, 3e-4, 12),
    
    # High frozen + very high unfrozen (pushing limits)
    ("frozen2e3_unfrozen2e4to1e3_12ep", 2e-3, 2e-4, 1e-3, 12),
]

# Fixed parameters
IMAGE_SIZE = 299
VALID_PCT = 0.2
WEIGHT_DECAY = 0.1
SEED = 42


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={
        str(REMOTE_DATA_DIR): data_volume,
        str(REMOTE_OUTPUT_DIR): results_volume,
    },
)
def train_config(
    config_name: str,
    lr_frozen: float,
    lr_unfrozen_min: float,
    lr_unfrozen_max: float,
    epochs: int,
    batch_size: int = 64,
    samples_per_class: int = 0,
):
    """Train EfficientNetV2-L gradual with specific LR schedule."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import models, transforms
    from PIL import Image
    from tqdm import tqdm
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("=" * 70)
    print(f"CONFIG: {config_name}")
    print("=" * 70)
    print(f"LR Frozen: {lr_frozen}")
    print(f"LR Unfrozen: {lr_unfrozen_min} -> {lr_unfrozen_max}")
    print(f"Epochs: {epochs}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Batch Size: {batch_size}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    device = "cuda"
    
    # Discover classes
    EXCLUDE_DIRS = {"__pycache__", ".git"}
    discovered_classes = []
    for d in sorted(REMOTE_DATA_DIR.iterdir()):
        if d.is_dir() and d.name not in EXCLUDE_DIRS:
            img_count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.JPG"))) + len(list(d.glob("*.png")))
            if img_count > 0:
                discovered_classes.append(d.name)
    
    CLASSES = discovered_classes
    print(f"Found {len(CLASSES)} classes")
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    train_tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    val_tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    class ArchDataset:
        def __init__(self, root, transform, classes):
            self.transform = transform
            self.classes = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.samples = []
            for cls_name in classes:
                cls_dir = root / cls_name
                if cls_dir.exists():
                    for img_path in cls_dir.iterdir():
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
            except:
                return self.__getitem__(random.randint(0, len(self) - 1))
    
    full_ds = ArchDataset(REMOTE_DATA_DIR, None, CLASSES)
    print(f"Total samples: {len(full_ds)}")
    
    if samples_per_class > 0:
        train_idx, val_idx = [], []
        for cls_idx in range(len(CLASSES)):
            indices = [i for i, (_, label) in enumerate(full_ds.samples) if label == cls_idx]
            random.shuffle(indices)
            n_train = min(samples_per_class, len(indices) // 2)
            n_val = min(samples_per_class, len(indices) - n_train)
            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
        random.shuffle(train_idx)
        random.shuffle(val_idx)
    else:
        indices = list(range(len(full_ds)))
        random.shuffle(indices)
        split = int(len(indices) * VALID_PCT)
        val_idx, train_idx = indices[:split], indices[split:]
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    train_ds = ArchDataset(REMOTE_DATA_DIR, train_tfm, CLASSES)
    val_ds = ArchDataset(REMOTE_DATA_DIR, val_tfm, CLASSES)
    
    effective_batch_size = min(batch_size, len(train_idx) // 2) if samples_per_class > 0 else batch_size
    effective_batch_size = max(effective_batch_size, 2)
    
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=effective_batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=effective_batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)
    
    if len(train_loader) == 0:
        raise ValueError("Need more samples for training")
    
    num_classes = len(CLASSES)
    model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Layer groups for gradual unfreezing
    layer_groups = {
        "head": ["classifier"],
        "features_late": ["features.7", "features.8"],
        "features_mid_late": ["features.5", "features.6"],
        "features_mid": ["features.3", "features.4"],
        "features_early": ["features.0", "features.1", "features.2"],
    }
    
    def freeze_all(model):
        for param in model.parameters():
            param.requires_grad = False
    
    def unfreeze_layer_group(model, prefixes):
        for name, param in model.named_parameters():
            if any(name.startswith(p + ".") or name == p for p in prefixes):
                param.requires_grad = True
    
    def get_param_groups(model, base_lr, lr_mult=0.5):
        groups = list(layer_groups.keys())
        param_groups = []
        assigned = set()
        for i, group_name in enumerate(groups):
            prefixes = layer_groups[group_name]
            params = []
            for n, p in model.named_parameters():
                if p.requires_grad and id(p) not in assigned:
                    if any(n.startswith(pfx + ".") or n == pfx for pfx in prefixes):
                        params.append(p)
                        assigned.add(id(p))
            if params:
                lr = base_lr * (lr_mult ** i)
                param_groups.append({"params": params, "lr": lr})
        remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned]
        if remaining:
            param_groups.append({"params": remaining, "lr": base_lr * (lr_mult ** len(groups))})
        return param_groups
    
    def train_epoch(model, loader, criterion, optimizer, scheduler):
        model.train()
        total_loss, correct, total = 0., 0, 0
        for images, labels in tqdm(loader, desc="Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        return total_loss / total, 100. * correct / total
    
    @torch.no_grad()
    def eval_epoch(model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0., 0, 0
        all_preds, all_labels = [], []
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return total_loss / total, 100. * correct / total, all_preds, all_labels
    
    # Gradual unfreezing schedule
    group_names = list(layer_groups.keys())
    epochs_per_phase = max(1, epochs // (len(group_names) + 1))
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_state = None
    current_epoch = 0
    
    for phase_idx in range(len(group_names) + 1):
        if phase_idx == 0:
            freeze_all(model)
            unfreeze_layer_group(model, layer_groups["head"])
            current_lr = lr_frozen
            print(f"\n=== Phase {phase_idx}: Head only (LR={current_lr}) ===")
        elif phase_idx < len(group_names):
            groups_to_unfreeze = group_names[:phase_idx + 1]
            for g in groups_to_unfreeze:
                unfreeze_layer_group(model, layer_groups[g])
            progress = phase_idx / len(group_names)
            current_lr = lr_unfrozen_min + progress * (lr_unfrozen_max - lr_unfrozen_min)
            print(f"\n=== Phase {phase_idx}: Unfreeze {groups_to_unfreeze[-1]} (LR={current_lr:.2e}) ===")
        else:
            for g in group_names:
                unfreeze_layer_group(model, layer_groups[g])
            current_lr = lr_unfrozen_max
            print(f"\n=== Phase {phase_idx}: All layers (LR={current_lr}) ===")
        
        param_groups = get_param_groups(model, current_lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        
        phase_epochs = epochs_per_phase if phase_idx < len(group_names) else (epochs - current_epoch)
        if phase_epochs <= 0:
            continue
            
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in param_groups],
            epochs=phase_epochs,
            steps_per_epoch=len(train_loader),
        )
        
        for epoch in range(phase_epochs):
            current_epoch += 1
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
            val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Epoch {current_epoch}/{epochs}: Train {train_acc:.1f}% | Val {val_acc:.1f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if current_epoch >= epochs:
                break
        if current_epoch >= epochs:
            break
    
    model.load_state_dict(best_state)
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion)
    
    output_dir = REMOTE_OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(best_state, output_dir / "model.pth")
    
    present_classes = sorted(set(final_labels) | set(final_preds))
    class_names = [CLASSES[i] for i in present_classes]
    report = classification_report(final_labels, final_preds, labels=present_classes, 
                                   target_names=class_names, digits=3)
    
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"Config: {config_name}\n")
        f.write(f"LR Frozen: {lr_frozen}\n")
        f.write(f"LR Unfrozen: {lr_unfrozen_min} -> {lr_unfrozen_max}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n\n")
        f.write(report)
    
    cm = confusion_matrix(final_labels, final_preds, labels=present_classes)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{config_name}\nAccuracy: {best_acc:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.suptitle(f'{config_name}')
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    
    results_volume.commit()
    
    print("\n" + "=" * 70)
    print(f"COMPLETED: {config_name}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print("=" * 70)
    
    return {
        "config_name": config_name,
        "lr_frozen": lr_frozen,
        "lr_unfrozen_min": lr_unfrozen_min,
        "lr_unfrozen_max": lr_unfrozen_max,
        "epochs": epochs,
        "best_acc": best_acc,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={
        str(REMOTE_DATA_DIR): data_volume,
        str(REMOTE_OUTPUT_DIR): results_volume,
    },
)
def train_resnet50_baseline(batch_size: int = 64, samples_per_class: int = 0):
    """Train ResNet50 baseline with standard 2-stage fine-tuning."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import models, transforms
    from PIL import Image
    from tqdm import tqdm
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    config_name = "resnet50_frozen1e3_unfrozen1e4_12ep"
    lr_frozen = 1e-3
    lr_unfrozen = 1e-4
    epochs_frozen = 6
    epochs_unfrozen = 6
    total_epochs = epochs_frozen + epochs_unfrozen
    
    print("=" * 70)
    print(f"RESNET50: {config_name}")
    print("=" * 70)
    print(f"LR Frozen: {lr_frozen}")
    print(f"LR Unfrozen: {lr_unfrozen}")
    print(f"Epochs: {epochs_frozen} + {epochs_unfrozen} = {total_epochs}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    device = "cuda"
    
    EXCLUDE_DIRS = {"__pycache__", ".git"}
    discovered_classes = []
    for d in sorted(REMOTE_DATA_DIR.iterdir()):
        if d.is_dir() and d.name not in EXCLUDE_DIRS:
            img_count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.JPG"))) + len(list(d.glob("*.png")))
            if img_count > 0:
                discovered_classes.append(d.name)
    
    CLASSES = discovered_classes
    print(f"Found {len(CLASSES)} classes")
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    train_tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    val_tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    class ArchDataset:
        def __init__(self, root, transform, classes):
            self.transform = transform
            self.classes = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.samples = []
            for cls_name in classes:
                cls_dir = root / cls_name
                if cls_dir.exists():
                    for img_path in cls_dir.iterdir():
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
            except:
                return self.__getitem__(random.randint(0, len(self) - 1))
    
    full_ds = ArchDataset(REMOTE_DATA_DIR, None, CLASSES)
    print(f"Total samples: {len(full_ds)}")
    
    if samples_per_class > 0:
        train_idx, val_idx = [], []
        for cls_idx in range(len(CLASSES)):
            indices = [i for i, (_, label) in enumerate(full_ds.samples) if label == cls_idx]
            random.shuffle(indices)
            n_train = min(samples_per_class, len(indices) // 2)
            n_val = min(samples_per_class, len(indices) - n_train)
            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
    else:
        indices = list(range(len(full_ds)))
        random.shuffle(indices)
        split = int(len(indices) * VALID_PCT)
        val_idx, train_idx = indices[:split], indices[split:]
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    train_ds = ArchDataset(REMOTE_DATA_DIR, train_tfm, CLASSES)
    val_ds = ArchDataset(REMOTE_DATA_DIR, val_tfm, CLASSES)
    
    effective_batch_size = min(batch_size, len(train_idx) // 2) if samples_per_class > 0 else batch_size
    
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=effective_batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=effective_batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)
    
    num_classes = len(CLASSES)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    def train_epoch(model, loader, criterion, optimizer, scheduler):
        model.train()
        total_loss, correct, total = 0., 0, 0
        for images, labels in tqdm(loader, desc="Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        return total_loss / total, 100. * correct / total
    
    @torch.no_grad()
    def eval_epoch(model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0., 0, 0
        all_preds, all_labels = [], []
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return total_loss / total, 100. * correct / total, all_preds, all_labels
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_state = None
    
    print("\n=== Phase 1: Frozen Backbone ===")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr_frozen, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_frozen, epochs=epochs_frozen, steps_per_epoch=len(train_loader)
    )
    
    for epoch in range(epochs_frozen):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs_frozen}: Train {train_acc:.1f}% | Val {val_acc:.1f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    print("\n=== Phase 2: Full Fine-tuning ===")
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_unfrozen, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_unfrozen, epochs=epochs_unfrozen, steps_per_epoch=len(train_loader)
    )
    
    for epoch in range(epochs_unfrozen):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epochs_frozen + epoch + 1}/{total_epochs}: Train {train_acc:.1f}% | Val {val_acc:.1f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion)
    
    output_dir = REMOTE_OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    present_classes = sorted(set(final_labels) | set(final_preds))
    class_names = [CLASSES[i] for i in present_classes]
    report = classification_report(final_labels, final_preds, labels=present_classes, 
                                   target_names=class_names, digits=3)
    
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"ResNet50: {config_name}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n\n")
        f.write(report)
    
    cm = confusion_matrix(final_labels, final_preds, labels=present_classes)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'ResNet50 - Accuracy: {best_acc:.2f}%')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    results_volume.commit()
    
    print("\n" + "=" * 70)
    print(f"COMPLETED: {config_name}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print("=" * 70)
    
    return {"config_name": config_name, "best_acc": best_acc}


@app.function(
    image=image,
    volumes={str(REMOTE_OUTPUT_DIR): results_volume},
    timeout=60 * 60,
)
def cleanup_non_best(best_config_name: str):
    """Delete model weights for non-best configurations."""
    print(f"Keeping weights for: {best_config_name}")
    
    for config_dir in REMOTE_OUTPUT_DIR.iterdir():
        if config_dir.is_dir() and config_dir.name != best_config_name:
            model_path = config_dir / "model.pth"
            if model_path.exists():
                model_path.unlink()
                print(f"  Deleted: {config_dir.name}/model.pth")
    
    results_volume.commit()
    print("Cleanup complete")


@app.local_entrypoint()
def main(tiny: bool = False, batch_size: int = 64):
    """Run LR sweep v2 - exploring around best configs from v1."""
    
    print("=" * 70)
    print("ARCHITECTURAL STYLE - LR SWEEP V2")
    print("Exploring around aggressive_unfreeze (75.4%) and high_everywhere (75.0%)")
    print("=" * 70)
    
    samples_per_class = 5 if tiny else 0
    if tiny:
        print("\n[TINY MODE] Using 5 samples per class")
    
    print("\nConfigurations:")
    for name, lr_f, lr_min, lr_max, ep in SWEEP_CONFIGS:
        print(f"  - {name}")
        print(f"      LR frozen={lr_f}, unfrozen={lr_min}->{lr_max}, epochs={ep}")
    
    print(f"\nSpawning {len(SWEEP_CONFIGS) + 1} training jobs...")
    
    handles = []
    
    # ResNet50 baseline
    handles.append(("resnet50", train_resnet50_baseline.spawn(
        batch_size=batch_size,
        samples_per_class=samples_per_class,
    )))
    
    # EfficientNetV2-L configs
    for name, lr_f, lr_min, lr_max, ep in SWEEP_CONFIGS:
        handles.append((name, train_config.spawn(
            config_name=name,
            lr_frozen=lr_f,
            lr_unfrozen_min=lr_min,
            lr_unfrozen_max=lr_max,
            epochs=ep,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
        )))
    
    results = []
    for name, h in handles:
        result = h.get()
        results.append(result)
        print(f"[OK] {result['config_name']}: {result['best_acc']:.2f}%")
    
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Rank':<6}{'Config':<50}{'Accuracy':>12}")
    print("-" * 70)
    for i, r in enumerate(results):
        marker = " [BEST]" if i == 0 else ""
        print(f"{i+1:<6}{r['config_name']:<50}{r['best_acc']:>11.2f}%{marker}")
    print("=" * 70)
    
    best_config = results[0]['config_name']
    print(f"\nCleaning up (keeping: {best_config})...")
    cleanup_non_best.remote(best_config)
    
    print("\n[OK] Complete!")
    print(f"Best: {best_config} with {results[0]['best_acc']:.2f}% accuracy")
    print("\nDownload results:")
    print("  modal volume get archstyle-xu2014-results-v2 / ./results_v2")
    
    return results

