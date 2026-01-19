# data loading and preprocessing functions
import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_data_from_csv(csv_path, img_dir):
    # load data from csv file
    df = pd.read_csv(csv_path)
    df['path'] = df['id_code'].apply(lambda x: os.path.join(img_dir, f'{x}.png'))
    df['target'] = df['diagnosis']
    return df


def ben_graham_processing(image):
    # ben graham preprocessing technique for retinal images
    img_size = 300
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        image = image[y:y+h, x:x+w]
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    mask = np.zeros((img_size, img_size), np.uint8)
    cv2.circle(mask, (img_size // 2, img_size // 2), img_size // 2, 1, thickness=-1)
    return cv2.bitwise_and(image, image, mask=mask)


class DRDataset(Dataset):
    # diabetic retinopathy dataset with ben graham preprocessing
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.df.iloc[idx]['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ben_graham_processing(img)
        label = torch.tensor(self.df.iloc[idx]['target'], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    # return train and validation transforms
    train_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_aug, val_aug


def create_dataloaders(train_df, val_df, batch_size=16):
    # create dataloaders for training and validation sets
    train_aug, val_aug = get_transforms()
    
    train_loader = DataLoader(
        DRDataset(train_df, train_aug),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        DRDataset(val_df, val_aug),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader
