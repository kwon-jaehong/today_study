import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from d_set import CustomDataset
from torch.utils.data import DataLoader, random_split
import pandas as pd


train_transform = A.Compose([
                        A.Resize(32,128),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])
# df,transforms
df = pd.read_csv('../train.csv')
data_path = "../"
train_dataset = CustomDataset(df,data_path,train_transform)
train_loader = DataLoader(train_dataset, batch_size = 32, num_workers = 4,pin_memory=True)

for image,label in train_loader:
    temp  = train_dataset.__getitem__(0)

print(0)