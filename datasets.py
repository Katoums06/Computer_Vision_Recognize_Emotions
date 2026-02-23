from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob 
import os
import cv2
import numpy as np
class Emotions_dataset(Dataset):
    def __init__(self, PATH, transform=None):

        self.PATH = PATH
        self.transform = transform

        self.samples = []
        self.len_class = []

        self.classes_to_idx = {
            'angry':0, 
            'disgust':1, 
            'fear':2, 
            'happy':3, 
            'neutral':4, 
            'sad':5, 
            'surprise':6
        }

        for class_, idx in self.classes_to_idx.items():
            folder_path = os.path.join(PATH, class_)
            files = glob.glob(os.path.join(folder_path, "*.*"))
            self.len_class.append(len(files))

            for file in files:
                self.samples.append((file, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        file, target = self.samples[index]
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, target
    


def show_img_dataset(dataloader : DataLoader):
    img, label = next(iter(dataloader))
    img = img.cpu().numpy() 
    label = label.cpu().numpy()
    row, col = dataloader.batch_size // 4, 4

    fig, axes = plt.subplots(row, col, figsize=(30, 30))
    axes = axes.flatten()

    for i in range(min(len(img), row*col)):
        img_ = img[i].squeeze(0)
        
        axes[i].imshow(img_, cmap='gray')
        axes[i].set_title(f"L: {label[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig) 