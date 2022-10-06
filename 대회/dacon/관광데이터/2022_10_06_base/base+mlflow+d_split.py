import random
import pandas as pd
import numpy as np
import os
import cv2


from sklearn.feature_extraction.text import CountVectorizer


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore')
import mlflow


os.chdir('../')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CFG = {
    'IMG_SIZE':128,
    'EPOCHS':5,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':64,
    'SEED':41,
    'TRAIN_RATE':0.9,
}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

all_df = pd.read_csv('./train.csv')
all_df_temp = all_df.copy()

lable_category_3 = list(set(list(all_df['cat3'])))
lable_category_3.sort()
label2num = {x:i for i,x in enumerate(lable_category_3)}
num2label = {i:x for i,x in enumerate(lable_category_3)}


def labeltonum(x,label2num):
    num = label2num[x['cat3']]
    return num
all_df['cat3'] = all_df.apply(labeltonum,args=(label2num,),axis=1)


vectorizer = CountVectorizer(max_features=4096)
all_vectors = vectorizer.fit_transform(all_df['overview'])
all_vectors = all_vectors.todense()

all_df['text_v'] = all_vectors.tolist()

class CustomDataset(Dataset):
    def __init__(self, df, transforms, infer=False):
        self.img_path_list = df['img_path'].to_list()
        self.text_vectors = df['text_v'].to_list()
        self.label_list = df['cat3'].to_list()
        
        
        self.transforms = transforms
        self.infer = infer
        
    def __getitem__(self, index):
        # NLP

        text_vector = self.text_vectors[index]
        
        # Image
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        # Label
        if self.infer:
            return image, torch.Tensor(text_vector).view(-1)
        else:
            label = self.label_list[index]
            return image, torch.Tensor(text_vector).view(-1), label
        
    def __len__(self):
        return len(self.img_path_list)


train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])


def split_df(df,train_rate,min_num,label_column):

    label_list = list(set(list(df[label_column])))
    label_list.sort()


    ## 각 항목별 df의 index가져옴
    train_index_list = []
    val_index_list = []
    for label in label_list:
        index_list = list(df[df[label_column]==label].index)
        ## 필요하다면 여기서 인덱스 리스트를 셔플해도 됨
        
        index_len = len(list(df[df[label_column]==label].index))

        if index_len*train_rate > min_num:
            train_index_list = train_index_list + index_list[:int(index_len*train_rate)]
            val_index_list = val_index_list + index_list[int(index_len*train_rate):]

    train_df = df.iloc[train_index_list]

    val_df = df.iloc[val_index_list]

    return train_df,val_df
    
train_df,val_df = split_df(all_df,CFG['TRAIN_RATE'],1,"cat3")

train_dataset = CustomDataset(train_df, train_transform)
validation_dataset = CustomDataset(val_df, train_transform)


train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=4)

val_loader = DataLoader(validation_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        # Image
        self.cnn_extract = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Text
        self.nlp_extract = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(4160, num_classes)
        )
            

    def forward(self, img, text):
        img_feature = self.cnn_extract(img)
        img_feature = torch.flatten(img_feature, start_dim=1)
        text_feature = self.nlp_extract(text)
        feature = torch.cat([img_feature, text_feature], axis=1)
        output = self.classifier(feature)
        return output


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0
    best_model = None
    
    for epoch in range(1,CFG["EPOCHS"]+1):
        model.train()
        train_loss = []
        for i,(img, text, label) in enumerate(train_loader):

            img = img.float().to(device)
            text = text.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()

            model_pred = model(img, text)
            
            loss = criterion(model_pred, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)
            
        val_loss, val_score = validation(model, criterion, val_loader, device)
        
        mlflow.log_metric("val_score",val_score)
        mlflow.log_metric("val_loss",val_loss)
        mlflow.log_metric("tr_loss",tr_loss)
        
            
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val Score : [{val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_score < val_score:
            best_score = val_score
            best_model = model
    
    mlflow.log_param("best_socre",best_score)

    return best_model


def score_function(real, pred):
    return f1_score(real, pred, average="weighted")
#real
# [118, 118, 118, 63, 118, 85, 44, 118, 118, 121, 73, 20, 118, 86, ...]
# pred
# [118, 118, 118, 90, 41, 85, 44, 118, 118, 121, 73, 20, 118, 44, ...]

def validation(model, criterion, val_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for img, text, label in tqdm(iter(val_loader)):
            img = img.float().to(device)
            text = text.to(device)
            label = label.to(device)
            
            model_pred = model(img, text)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    test_weighted_f1 = score_function(true_labels, model_preds)
    return np.mean(val_loss), test_weighted_f1


model = CustomModel(len(label2num))
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None


with mlflow.start_run() as run:
    mlflow.log_artifact(__file__)
    infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)




