from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import cv2
    
class CustomDataset(Dataset):
    def __init__(self,df,data_path,transforms,infer=False):
     
        self.df = df  
        self.transforms = transforms
        self.infer = infer
        
        self.id = list(df['id'])
        self.img_path = list(df['img_path'])
        self.label = list(df['label'])
        self.data_path = data_path
        
        
    def __getitem__(self, index):
        image = cv2.imread("."+self.img_path[index])
        label = self.label[index]

        if self.transforms is not None:
            image = self.transforms(image=image)['image']        
        # Label
        if self.infer:
            return image
        else:
            return image,label

    def __len__(self):
        return len(self.df)