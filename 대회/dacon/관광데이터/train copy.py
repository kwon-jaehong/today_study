import random
import pandas as pd
import numpy as np
import os
import cv2
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import f1_score
import warnings
from konlpy.tag import Mecab
from torchtext import data

warnings.filterwarnings(action='ignore')
CFG = {
    'IMG_SIZE':128,
    'EPOCHS':5,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':64,
    'SEED':41
}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

seed_everything(CFG['SEED']) # Seed 고정


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


all_df = pd.read_csv('./train.csv')
train_df, val_df, _, _ = train_test_split(all_df, all_df['cat3'], test_size=0.2, random_state=CFG['SEED'])

le = preprocessing.LabelEncoder()
le.fit(train_df['cat3'].values)


train_df['cat3'] = le.transform(train_df['cat3'].values)
val_df['cat3'] = le.transform(val_df['cat3'].values)


train_df.to_csv('./temp.csv',index=False)
val_df.to_csv('./temp2.csv',index=False)


mecab = Mecab()
TEXT = data.Field(tokenize = mecab.morphs, preprocessing = generate_bigrams)
LABEL = data.LabelField(dtype = torch.float)
fields = {'overview': ('text',TEXT), 'cat3': ('label',LABEL)}



train_data, test_data = data.TabularDataset.splits(
                            path = './',
                            train = 'temp.csv',
                            test = 'temp2.csv',
                            format = 'csv',
                            fields = fields,
)

# vars(train_data[0])
# {'text': ['함평', '양서', '파충류', '생태', '공원', '은', '전남', '함평군', '신광면', ...], 'label': '44'}
# <class 'dict'>


MAX_VOCAB_SIZE = 100000


## CustomDataset 선언할때 넣어야함
## 최대 vocab size = 583169
TEXT.build_vocab(train_data,max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)
train_loader = data.Iterator(dataset=train_data, batch_size = 2000)

temp= 1

class CustomDataset(Dataset):
    def __init__(self, img_path_list, text_vectors, label_list, transforms, infer=False):
        self.img_path_list = img_path_list
        self.text_vectors = text_vectors
        self.label_list = label_list
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
train_dataset = CustomDataset(train_df['img_path'].values, train_vectors, train_df['cat3'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=6)
val_dataset = CustomDataset(val_df['img_path'].values, val_vectors, val_df['cat3'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=6)



class CustomModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
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
        for img, text, label in tqdm(iter(train_loader)):
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
            
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val Score : [{val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_score < val_score:
            best_score = val_score
            best_model = model
    
    return best_model





def score_function(real, pred):
    return f1_score(real, pred, average="weighted")

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



model = CustomModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test_df = pd.read_csv('./test.csv')
test_vectors = vectorizer.transform(test_df['overview'])
test_vectors = test_vectors.todense()


test_dataset = CustomDataset(test_df['img_path'].values, test_vectors, None, test_transform, True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=6)

def inference(model, test_loader, deivce):
    model.to(device)
    model.eval()
    
    model_preds = []
    
    with torch.no_grad():
        for img, text in tqdm(iter(test_loader)):
            img = img.float().to(device)
            text = text.to(device)
            
            model_pred = model(img, text)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    
    return model_preds



preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')
submit['cat3'] = le.inverse_transform(preds)

submit.to_csv('./submit.csv', index=False)





























