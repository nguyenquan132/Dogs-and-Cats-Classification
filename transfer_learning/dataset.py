import os
import cv2
from pathlib import Path
from torch.utils.data import Dataset

class CatDogDataset(Dataset): 
    def __init__(self, rootDir, label2id, transforms=None): 
        assert isinstance(rootDir, str)
        self.rootDir = rootDir
        self.label2id = label2id
        self.transforms = transforms
        subFolder = os.listdir(rootDir)
        if Path(os.path.join(rootDir, subFolder[0])).is_dir(): 
            self.list_images = []
            for folder in subFolder: 
                folder_path = os.path.join(rootDir, folder)
                for img in os.listdir(folder_path): 
                    img_path = os.path.join(folder_path, img)
                    self.list_images.append(img_path)
    def __len__(self): 
        return len(self.list_images)
    def __getitem__(self, idx): 
        image_id = self.list_images[idx]
        label_name = Path(image_id).parent.name
        label_id = self.label2id[label_name] 
        image = cv2.imread(image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms: 
            augmented = self.transforms(image=image)
            image = augmented['image']
        return image, label_id