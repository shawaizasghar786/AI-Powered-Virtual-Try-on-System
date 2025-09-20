import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# configurations
IMG_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\train\cloth'
MASK_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\train\cloth-mask'
TEST_IMG_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\test\cloth'
TEST_MASK_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\test\cloth-mask'
BATCH_SIZE = 8
NUM_EPOCHS = 75
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'new_segmentation_unet.pth'
IMG_SIZE = (256, 192)
WEIGHT_DECAY = 1e-5
DROPOUT_PROB = 0.1
VAL_SPLIT = 0.2  

# dataset class
class ClothSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, img_names, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.img_names = img_names
        self.augment = augment
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.albumentations_transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.05, p=0.3),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.augment:
            augmented = self.albumentations_transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image'].float() / 255.0  # Normalize to [0,1]
            image = self.normalize(image)  # Apply ImageNet normalization
            mask = augmented['mask'].unsqueeze(0).float()
        else:
            image = self.transform(image).float()
            mask = self.mask_transform(mask).float()

        # Ensure mask is binary
        mask = (mask > 0.5).float()
        return image, mask


# model (U-Net)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_prob=0.1):
        super(UNet, self).__init__()
        self.dropout_prob = dropout_prob

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.middle = nn.Sequential(
            CBR(512, 1024),
            nn.Dropout(dropout_prob)
        )
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m = self.middle(self.pool(e4))
        d4 = self.up4(m)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return torch.sigmoid(out)


# --- TRAINING ---
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device).float()
        masks = masks.to(device).float()
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# --- VALIDATION ---
def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device).float()
            masks = masks.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# --- TESTING ---
def test(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_dataloader, desc="Testing"):
            images = images.to(device).float()
            masks = masks.to(device).float()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            # Calculate Dice score for segmentation quality
            pred = (outputs > 0.5).float()
            intersection = (pred * masks).sum((1,2,3))
            union = pred.sum((1,2,3)) + masks.sum((1,2,3))
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_scores.extend(dice.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_dataloader)
    avg_dice_score = np.mean(dice_scores)
    
    return avg_test_loss, avg_dice_score


def main():
    # Get all image names
    all_img_names = sorted([f for f in os.listdir(IMG_DIR) if os.path.isfile(os.path.join(IMG_DIR, f))])
    num_total = len(all_img_names)
    indices = np.arange(num_total)
    np.random.shuffle(indices)

    # Split indices for training and validation
    val_size = int(VAL_SPLIT * num_total)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_img_names = [all_img_names[i] for i in train_indices]
    val_img_names = [all_img_names[i] for i in val_indices]

    # Create Datasets and DataLoaders
    train_dataset = ClothSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, img_names=train_img_names, augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = ClothSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, img_names=val_img_names, augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_dataset = ClothSegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR, IMG_SIZE,
                                             img_names=sorted([f for f in os.listdir(TEST_IMG_DIR) if
                                                               os.path.isfile(os.path.join(TEST_IMG_DIR, f))]),
                                             augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    model = UNet(dropout_prob=DROPOUT_PROB).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Training on: {DEVICE}")

    best_val_loss = float('inf')
    train_losses = []  # Track training losses
    val_losses = []    # Track validation losses
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_dataloader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_dataloader, criterion, DEVICE)
        
        # Store the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation loss improved, saving model to {MODEL_SAVE_PATH}")

    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.close()
    
    print("Learning curves have been saved to 'learning_curves.png'")


if __name__ == "__main__":
    main()