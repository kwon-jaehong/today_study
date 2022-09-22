from distutils.command.config import config
import random
import pandas as pd
import numpy as np
import os
import cv2
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

# 토치 텍스트 관련 참조자료
# https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e


warnings.filterwarnings(action='ignore')
CFG = {
    'IMG_SIZE':128,
    'EPOCHS':1000,
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


seed_everything(CFG['SEED']) # Seed 고정


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')





class Vocabulary:
    def __init__(self, freq_threshold, max_size,mecab):
        self.itos = {0: '<unk>', 1:'<pad>'}
        self.stoi = {k:j for j,k in self.itos.items()} 
        self.mecab = mecab
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        
        self.bi_gram =True
    
    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 2
        self.max_len = 0

        
        for sentence in sentence_list:
            sentence = self.mecab.morphs(sentence)
            
            ## 바이그램으로 구성할지 여부
            if self.bi_gram:
                sentence = self.generate_bigrams(sentence)
            
            ## 문장 최대 길이
            if len(sentence) > self.max_len:
                self.max_len = len(sentence)
            
            for word in sentence:
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
        frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold} 
        frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-idx]) # idx =4 for pad, start, end , unk
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1
            

        

    def numericalize(self, text):
        tokenized_text = self.mecab.morphs(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else:
                numericalized_text.append(self.stoi['<unk>'])
                
        return numericalized_text
    
    ## bi그램 처리
    def generate_bigrams(self,x):
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

    


class CustomDataset(Dataset):
    def __init__(self,csv_path,max_vocab_size,transforms,infer=False):
        self.all_df = pd.read_csv(csv_path)        
        
        ## 텍스트 리스트에서 html문법 빼기
        self.text_list = list(self.all_df['overview'])
        self.img_path_list = list(self.all_df['img_path'])
        
        self.label_list = list(self.all_df['cat3'])
        self.num2label = {i:label for i,label in enumerate(list(set(self.label_list)))}
        self.lable2num = {label:i for i,label in enumerate(list(set(self.label_list)))}
        
        self.TEXT = Vocabulary(0,max_vocab_size,Mecab())
        self.TEXT.build_vocabulary(self.text_list)
        
        

                
        # self.TEXT.numericalize("소년는 나를")
        

        self.transforms = transforms
        self.infer = infer
        
    def __getitem__(self, index):
       
        text = self.text_list[index]
        text_vector = self.TEXT.numericalize(text)
        text_len = len(text_vector)
        
        ## Image
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        # Label
        if self.infer:
            return image, torch.Tensor(text_vector).view(-1)
        else:
            label = self.lable2num[self.label_list[index]]
            return image, torch.Tensor(text_vector).view(-1).to(torch.long), torch.tensor([label],dtype=torch.long),torch.tensor([text_len],dtype=torch.long)
        

    
    def __len__(self):
        return len(self.all_df)
          



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
MAX_VOCAB_SIZE = 100000


print("데이터셋 구성 중")
dataset_temp = CustomDataset('./train.csv',MAX_VOCAB_SIZE,train_transform)
print("데이터셋 구성 완료")

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
        text = [item[1] for item in batch]  
        text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value = self.pad_idx) 
        
        image = [item[0] for item in batch]  
        label = [item[2] for item in batch]  
        text_len = [item[3] for item in batch]  

        return {"image":torch.stack(image),"text":text,"label":torch.stack(label).squeeze(),"text_len":torch.stack(text_len)}
    
def get_train_loader(dataset, batch_size, num_workers=0, pin_memory=True):
    pad_idx = dataset.TEXT.stoi['<pad>']
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader





    

class CustomModel(nn.Module):
    def __init__(self,vocab_size, num_classes):
        super(CustomModel, self).__init__()
        
        self.embedding_dim = 1024
        self.vocab_size = vocab_size
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
        
        # 이미지만 3136
        
        # Text
        self.bon_embed = nn.Embedding(vocab_size,self.embedding_dim,padding_idx=1)
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.nlp_bn = nn.BatchNorm1d(self.embedding_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(4160, num_classes)
        )
            

    def forward(self, img, text,text_len):
        img_feature = self.cnn_extract(img)
        img_feature = torch.flatten(img_feature, start_dim=1)
        
        embed = self.bon_embed(text)
        embed = torch.sum(embed, 1).squeeze(1)
        batch_size = embed.size(0)
        # text_len = text_len.float().unsqueeze(1)
        text_len = text_len.expand(batch_size, self.embedding_dim)
        embed /= text_len
        embed = self.nlp_bn(embed)
        text_feature = self.fc(embed)
        
        feature = torch.cat([img_feature, text_feature], axis=1)
        
        
        output = self.classifier(feature)
        return output

print("데이터 로드중 완료")
train_loader = get_train_loader(dataset_temp, CFG['BATCH_SIZE'])
print("데이터 로드 완료")
model = CustomModel(len(dataset_temp.TEXT.stoi),len(dataset_temp.lable2num))



model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
best_score = 0
best_model = None
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])

data_len = dataset_temp.__len__()

for epoch in range(1,CFG["EPOCHS"]+1):
    model.train()
    train_loss = []
    
    
    total_correct = 0
    
    for i,data_batch in enumerate(train_loader):
        img = data_batch['image']
        text = data_batch['text']
        label = data_batch['label']
        text_len = data_batch['text_len']
        
        img = img.float().to(device)
        text = text.to(device)
        label = label.to(device)
        text_len = text_len.to(device)
        optimizer.zero_grad()
        
        model_pred = model(img, text,text_len)
        
        loss = criterion(model_pred, label)
        loss.backward()
        optimizer.step()
        
        
        _, predicted = torch.max(model_pred, 1) 
        correct = (predicted == label).sum().item()
        
        total_correct += correct
        
        train_loss.append(loss.item())
        
        print(f'epoch {epoch}/{CFG["EPOCHS"]+1} {(i*CFG["BATCH_SIZE"])+len(label)}/{data_len} train loss {loss.item():.4f} correct : {100*correct/len(label):.2f}% - ({correct}/{len(label)})')
        
    tr_loss = np.mean(train_loss)
    print(f"\n epoch {epoch} end!!! \t train loss : {tr_loss}\t total acc : {100*total_correct/data_len:.2f}% - ({total_correct}/{data_len}) \n")

# def train(model, optimizer, train_loader, val_loader, scheduler, device):

#         for img, text, label in tqdm(iter(train_loader)):
#             img = img.float().to(device)
#             text = text.to(device)
#             label = label.to(device)
            
#             optimizer.zero_grad()

#             model_pred = model(img, text)
            
#             loss = criterion(model_pred, label)

#             loss.backward()
#             optimizer.step()

#             train_loss.append(loss.item())

#         tr_loss = np.mean(train_loss)
            
#         val_loss, val_score = validation(model, criterion, val_loader, device)
            
#         print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val Score : [{val_score:.5f}]')
        
#         if scheduler is not None:
#             scheduler.step()
            
#         if best_score < val_score:
#             best_score = val_score
#             best_model = model
    
#     return best_model





# def score_function(real, pred):
#     return f1_score(real, pred, average="weighted")

# def validation(model, criterion, val_loader, device):
#     model.eval()
    
#     model_preds = []
#     true_labels = []
    
#     val_loss = []
    
#     with torch.no_grad():
#         for img, text, label in tqdm(iter(val_loader)):
#             img = img.float().to(device)
#             text = text.to(device)
#             label = label.to(device)
            
#             model_pred = model(img, text)
            
#             loss = criterion(model_pred, label)
            
#             val_loss.append(loss.item())
            
#             model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
#             true_labels += label.detach().cpu().numpy().tolist()
        
#     test_weighted_f1 = score_function(true_labels, model_preds)
#     return np.mean(val_loss), test_weighted_f1



# model = CustomModel()
# model.eval()
# optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
# scheduler = None

# infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

# test_df = pd.read_csv('./test.csv')
# test_vectors = vectorizer.transform(test_df['overview'])
# test_vectors = test_vectors.todense()


# test_dataset = CustomDataset(test_df['img_path'].values, test_vectors, None, test_transform, True)
# test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=6)

# def inference(model, test_loader, deivce):
#     model.to(device)
#     model.eval()
    
#     model_preds = []
    
#     with torch.no_grad():
#         for img, text in tqdm(iter(test_loader)):
#             img = img.float().to(device)
#             text = text.to(device)
            
#             model_pred = model(img, text)
#             model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    
#     return model_preds



# preds = inference(infer_model, test_loader, device)

# submit = pd.read_csv('./sample_submission.csv')
# submit['cat3'] = le.inverse_transform(preds)

# submit.to_csv('./submit.csv', index=False)





























