import random
import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import f1_score
from konlpy.tag import Mecab

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# 토치 텍스트 관련 참조자료
# https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e


CFG = {
    'IMG_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':64,
    'SEED':41,
    'MAX_VOCAB_SIZE':100000,
    'TRAIN_RATE':0.9,
    'NUM_WORKERS':4
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
        self.infer = infer
        if infer ==False:
            self.label_list = list(self.all_df['cat3'])
            self.num2label = {i:label for i,label in enumerate(list(set(self.label_list)))}
            self.lable2num = {label:i for i,label in enumerate(list(set(self.label_list)))}
        
        self.TEXT = Vocabulary(0,max_vocab_size,Mecab())
        self.TEXT.build_vocabulary(self.text_list)        

        self.transforms = transforms
        
        
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
            return image, torch.Tensor(text_vector).view(-1),torch.tensor([text_len],dtype=torch.long)
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


print("데이터셋 구성 중")
train_all_dataset = CustomDataset('./train.csv',CFG['MAX_VOCAB_SIZE'],train_transform)
vocab_size = len(train_all_dataset.TEXT.itos)
label_info = train_all_dataset.lable2num
print("데이터셋 구성 완료")

dataset_size = len(train_all_dataset)
train_size = int(dataset_size * CFG['TRAIN_RATE'])
validation_size = dataset_size - train_size

train_dataset, validation_dataset = random_split(train_all_dataset, [train_size, validation_size])



class MyCollate:
    def __init__(self, pad_idx,infer=False):
        self.pad_idx = pad_idx
        self.infer = infer

    def __call__(self, batch):
        image = [item[0] for item in batch]  
        text = [item[1] for item in batch]  
        text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value = self.pad_idx) 
        if self.infer == False:
            label = [item[2] for item in batch]  
            text_len = [item[3] for item in batch]  
            return {"image":torch.stack(image),"text":text,"label":torch.stack(label).squeeze(),"text_len":torch.stack(text_len)}
        else:
            text_len = [item[2] for item in batch]  
            return {"image":torch.stack(image),"text":text,"text_len":torch.stack(text_len)}

pad_idx = train_all_dataset.TEXT.stoi['<pad>']
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_idx))
validation_loader = DataLoader(validation_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_idx))
    

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


model = CustomModel(vocab_size,len(label_info))
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
best_score = 0
best_model = None
# optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"],weight_decay=1e-4)
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
# 손실함수에 어떤 제약 조건을 적용해 오버피팅을 최소화하는 방법으로 L1 정형화와 L2 정형화가 있습니다. 오버피팅은 특정 가중치값이 커질수록 발생할 가능성이 높아지므로 이를 해소하기 위해 특정값을 손실함수에 더해주는 것이 정형화 중 가중치 감소(Weight Decay)이며, 더해주는 특정값을 결정하는 것이 L1 정형화와 L2 정형화입니다. 파이토치에서 이 Weight Decay는 다음 코드처럼 적용할 수 있습니다.
# 결과적으로 weight_decay의 값이 커질수록 가중치 값이 작어지게 되고, 오버피팅 현상을 해소할 수 있지만, weight_decay 값을 너무 크게 하면 언더피팅 현상이 발생하므로 적당한 값을 사용해야 합니다.

def score_function(real, pred):
    return f1_score(real, pred, average="weighted")


for epoch in range(1,CFG["EPOCHS"]+1):
    model.train()
    train_loss = []   
    train_data_len = train_dataset.__len__()
    total_train_correct = 0
    
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
        total_train_correct += correct        
        train_loss.append(loss.item())
        
        print(f'epoch {epoch}/{CFG["EPOCHS"]+1} {(i*CFG["BATCH_SIZE"])+len(label)}/{train_data_len} train loss {loss.item():.4f} acc : {100*correct/len(label):.2f}% - ({correct}/{len(label)})')
        
    tr_loss = np.mean(train_loss)
    print(f"\n epoch {epoch} train end!!! \t train batch loss : {tr_loss:.4f}\t total acc : {100*total_train_correct/train_data_len:.2f}% - ({total_train_correct}/{train_data_len}) \n")
    
    

    
    
    val_data_len = validation_dataset.__len__()
    model.eval()   
    model_preds = []
    true_labels = []    
    val_loss = []    
    total_val_correct = 0
    with torch.no_grad():
        for img, text, label,text_len in validation_loader:
            img = data_batch['image']
            text = data_batch['text']
            label = data_batch['label']
            text_len = data_batch['text_len']
        
            img = img.float().to(device)
            text = text.to(device)
            label = label.to(device)
            text_len = text_len.to(device)
            
            model_pred = model(img, text,text_len)
            
            loss = criterion(model_pred, label)
            
            _, predicted = torch.max(model_pred, 1) 
            correct = (predicted == label).sum().item() 
            total_val_correct+=correct
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    test_weighted_f1 = score_function(true_labels, model_preds)
    
    print(f"epoch {epoch} val end!!! val loss : {np.mean(val_loss):.3f} \t f1 score : {test_weighted_f1:.3f} acc : {100*total_val_correct/val_data_len:.2f}% - ({total_val_correct}/{val_data_len}) \n\n")    

    
    writer.add_scalars("loss",{"tr_loss":tr_loss,"val loss":np.mean(val_loss)},epoch)
    writer.add_scalars("acc",{"tr_acc":total_train_correct/train_data_len,"val_acc":total_val_correct/val_data_len},epoch)
    torch.save(model.state_dict(),'./'+str(epoch)+".pth")
    



test_dataset = CustomDataset('./test.csv',CFG['MAX_VOCAB_SIZE'],test_transform,infer=True)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_idx,infer=True))
    
model.to(device)
model.eval()

model_preds = []

with torch.no_grad():
    for img, text, text_len in test_loader:
        img = data_batch['image']
        text = data_batch['text']

        text_len = data_batch['text_len']
    
        img = img.float().to(device)
        text = text.to(device)

        text_len = text_len.to(device)
        
        model_pred = model(img, text,text_len)
        model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()



submit = pd.read_csv('./sample_submission.csv')
submit['cat3'] = "dd"

submit.to_csv('./submit.csv', index=False)





























