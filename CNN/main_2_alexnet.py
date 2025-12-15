
import sys
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.generic_image_dataset import GenericImageDataset
from torch.optim.lr_scheduler import StepLR
from lib.dataset_precreator import get_images_by_class, split_data
from lib.metrics import (
    evaluate_full, calculate_metrics,
    plot_confusion_matrix, plot_roc, plot_metric_curves, plot_learning_rate_curve
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from CNN.Models.conv_mixer_classifier import ConvMixerClassifier
from CNN.Models.alexnet_classifier import AlexNetClassifier
from lib.print_dataset_distribution import print_dataset_distribution
from CNN.Models.alexnet_overrided import AlexNetOverrided

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience  # Kaç epoch bekleyecek
        self.delta = delta        # En az ne kadar iyileşme sayılır
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"⏳ EarlyStopping count: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing  # 🧠 işte burada!
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



def save_model(model, optimizer, epoch, path="convmixer_model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"✅ Model başarıyla kaydedildi: {path}")
def evaluate_val_loss(model, dataloader, loss_fn, device):
    

    model.eval()
    batch_losses = []
    debug_done = False

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            batch_losses.append(loss.item())  # Artık çarpma yok

            # İlk batch için debug
            if not debug_done:
                print("🧪 Sample raw model output range:")
                print("   min:", outputs.min().item(), "max:", outputs.max().item())

                softmax_out = torch.softmax(outputs, dim=1)
                print("🧪 Softmax range:")
                print("   min:", softmax_out.min().item(), "max:", softmax_out.max().item())
                print("   sample prediction:", torch.argmax(softmax_out, dim=1)[:5].tolist())
                debug_done = True

    avg_loss = sum(batch_losses) / len(batch_losses)
    print(f"✅ Val Loss (avg): {avg_loss:.4f}")
    return avg_loss

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    # Resize all images to 140x100 (minimum size of images)

    
    set_seed(42)
    resize_dim = (224, 224)
    train_transform = transforms.Compose([
    transforms.Resize(resize_dim),
    transforms.RandomHorizontalFlip(p=0.5),      # Yatay çevir
    transforms.RandomRotation(10),               # +/- 10 derece döndür
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Renk varyasyonu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # AlexNet için normalize
                         std=[0.229, 0.224, 0.225])
])
    image_dict, class_names = get_images_by_class("Datasets", max_per_class=837)

    train_data, val_data, test_data = split_data(image_dict, class_names)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    print("Sınıf numaraları:", class_to_idx)
    
    print( "📊 Dataset sınıf dağılımı:" )
    
    # print(print_dataset_distribution(val_data, class_names))
    

    sys.exit(0)

    
    train_dataset = GenericImageDataset(train_data, resize_dim=resize_dim, transform=train_transform)
    val_dataset   = GenericImageDataset(val_data, resize_dim=resize_dim)
    test_dataset  = GenericImageDataset(test_data, resize_dim=resize_dim)
 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32)
    test_loader  = DataLoader(test_dataset, batch_size=32)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    
    # model = AlexNetClassifier(num_classes=len(class_names) ,pretrained=True).to(device)
    model = AlexNetOverrided(num_classes=len(class_names), pretrained=False).to(device)
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn = FocalLoss(gamma=1.5, label_smoothing=0.1)  
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=10, delta=0.01)

    epochs = 50
    metrics_log = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_specificity': [], 'val_specificity': [],
        'train_sensitivity': [], 'val_sensitivity': [],
        'train_f1_score': [], 'val_f1_score': [],
        'learning_rate': []  

    }

    def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
        model.train()
        total_loss, correct = 0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
        return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")  # bu çıktıyı mutlaka al

        train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # Train metrikleri
        y_true_train, y_pred_train, y_proba_train = evaluate_full(model, train_loader, device)
        train_metrics = calculate_metrics(y_true_train, y_pred_train, y_proba_train)

        # Val metrikleri
        val_loss = evaluate_val_loss(model, val_loader, loss_fn, device)


        y_true_val, y_pred_val, y_proba_val = evaluate_full(model, val_loader, device)
        val_metrics = calculate_metrics(y_true_val, y_pred_val, y_proba_val)
        metrics_log['train_loss'].append(train_loss)
        metrics_log['val_loss'].append(val_loss)

        # Loglama
        
        for key in ['accuracy', 'f1_score', 'specificity', 'sensitivity']:
            metrics_log[f'train_{key}'].append(train_metrics[key])
            metrics_log[f'val_{key}'].append(val_metrics[key])

        print(f"[{epoch+1}/{epochs}] TL: {train_loss:.4f} | TA: {train_metrics['accuracy']:.4f} | "
            f"VL: {val_loss:.4f} | VA: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f}")

        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(f"🔁 Current LR: {current_lr:.6f}")
            metrics_log['learning_rate'].append(current_lr)
        early_stopping(val_loss)

        if early_stopping.early_stop:
                break


    # 4. Eğitim Sonrası Görseller
    plot_metric_curves(metrics_log)
    plot_confusion_matrix(val_metrics['confusion_matrix'], class_names)
    plot_roc(y_true_val, y_proba_val, class_names)
    plot_learning_rate_curve(metrics_log)

    # 5. Test Set Değerlendirmesi
    y_true_test, y_pred_test, y_proba_test = evaluate_full(model, test_loader, device)
    test_metrics = calculate_metrics(y_true_test, y_pred_test, y_proba_test)

    print("\n[TEST SET]")
    for key, val in test_metrics.items():
        if key != 'confusion_matrix':
            print(f"{key}: {val:.4f}")

    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, title="Test Confusion Matrix")
    plot_roc(y_true_test, y_proba_test, class_names)
    save_model(model, optimizer, epoch=epochs, path="CNN/Outputs/alexnet_model11.pth")

