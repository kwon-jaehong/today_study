import random
import torch
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset import CustomDataset,MyCollate
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model import CustomModel
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
from torch.optim.lr_scheduler import LambdaLR
import logging as log
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, Sequence


log.basicConfig(filename='./log.txt', level=log.DEBUG)

writer = SummaryWriter('runs/experiment_1')
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        if len(target.shape) ==0:
            target = target.view(-1)
            
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return focal_loss
    
    
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10,
    'LEARNING_RATE':2e-5,
    'BATCH_SIZE':6,
    'SEED':41,
    'TRAIN_RATE':0.9,
    'NUM_WORKERS':4
}
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

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


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

print("데이터셋 구성 중")
train_all_dataset = CustomDataset('./train.csv',tokenizer,train_transform)
label_info = train_all_dataset.label_3_level_2num
print("데이터셋 구성 완료")

dataset_size = len(train_all_dataset)
train_size = int(dataset_size * CFG['TRAIN_RATE'])
validation_size = dataset_size - train_size

## 데이터셋 나누기
train_dataset, validation_dataset = random_split(train_all_dataset, [train_size, validation_size])

## 데이터 로더
pad_token_id = tokenizer.pad_token_id
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id))
validation_loader = DataLoader(validation_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id))
    
    
    
model = CustomModel(tokenizer,len(train_all_dataset.label_1_level_2num),len(train_all_dataset.label_2_level_2num),len(label_info))
model.to(device)
criterion = FocalLoss().to(device)

# criterion = nn.CrossEntropyLoss().to(device)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=CFG['LEARNING_RATE'], eps=1e-08)
    
## 학습할 최종 스텝 계산    
t_total = len(train_loader) * CFG['EPOCHS']
warmup_steps = 0
# lr 조금씩 감소시키는 스케줄러
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)




for epoch in range(1,CFG["EPOCHS"]+1):
    model.train()
    train_loss = []   
    train_data_len = train_dataset.__len__()
    
    total_train_correct_cat1 = 0
    total_train_correct_cat2 = 0
    total_train_correct_cat3 = 0
    
    for i,data_batch in enumerate(train_loader):
        img = data_batch['image']
        text = data_batch['text']
        cat1 = data_batch['label_1']
        cat2 = data_batch['label_2']
        cat3 = data_batch['label_3']
        mask = data_batch['mask']
        
        
        img = img.float().to(device)
        text = text.to(device)
        mask = mask.to(device)
        cat1 = cat1.to(device)
        cat2 = cat2.to(device)
        cat3 = cat3.to(device)
        
        optimizer.zero_grad()
        
        cat1_p,cat2_p,cat3_p = model(img, text, mask, device)        
        
        loss1 = criterion(cat1_p,cat1)
        loss2 = criterion(cat2_p,cat2)
        loss3 = criterion(cat3_p,cat3)
        
        total_loss = loss1+loss2+loss3
        
        total_loss.backward()
        optimizer.step()                
        scheduler.step()
        
        _, predicted_1 = torch.max(cat1_p, 1) 
        correct_1 = (predicted_1 == cat1).sum().item()        
        total_train_correct_cat1 += correct_1        
        
        _, predicted_2 = torch.max(cat2_p, 1) 
        correct_2 = (predicted_2 == cat2).sum().item()        
        total_train_correct_cat2 += correct_2        
        
        _, predicted_3 = torch.max(cat3_p, 1) 
        correct_3 = (predicted_3 == cat3).sum().item()        
        total_train_correct_cat3 += correct_3        
        
        
        train_loss.append(total_loss.item())
        
        print(f'epoch {epoch}/{CFG["EPOCHS"]+1} {(i*CFG["BATCH_SIZE"])+len(cat3)}/{train_data_len} train loss {total_loss.item():.4f} cat1_acc : {100*correct_1/len(cat1):.2f}% - ({correct_1}/{len(cat1)}) cat2_acc : {100*correct_2/len(cat2):.2f}% - ({correct_2}/{len(cat2)}) cat3_acc : {100*correct_3/len(cat3):.2f}% - ({correct_3}/{len(cat3)})')
        log.info(f'epoch {epoch}/{CFG["EPOCHS"]+1} {(i*CFG["BATCH_SIZE"])+len(cat3)}/{train_data_len} train loss {total_loss.item():.4f} cat1_acc : {100*correct_1/len(cat1):.2f}% - ({correct_1}/{len(cat1)}) cat2_acc : {100*correct_2/len(cat2):.2f}% - ({correct_2}/{len(cat2)}) cat3_acc : {100*correct_3/len(cat3):.2f}% - ({correct_3}/{len(cat3)})')
    
    tr_loss = np.mean(train_loss)
    print(f"\n epoch {epoch} train end!!! \t train batch loss : {tr_loss:.4f}\t total_cat1_acc : {100*total_train_correct_cat1/train_data_len:.2f}% - ({total_train_correct_cat1}/{train_data_len}) total_cat2_acc : {100*total_train_correct_cat2/train_data_len:.2f}% - ({total_train_correct_cat2}/{train_data_len}) total_cat3_acc : {100*total_train_correct_cat3/train_data_len:.2f}% - ({total_train_correct_cat3}/{train_data_len}) \n")
    log.info(f"\n epoch {epoch} train end!!! \t train batch loss : {tr_loss:.4f}\t total_cat1_acc : {100*total_train_correct_cat1/train_data_len:.2f}% - ({total_train_correct_cat1}/{train_data_len}) total_cat2_acc : {100*total_train_correct_cat2/train_data_len:.2f}% - ({total_train_correct_cat2}/{train_data_len}) total_cat3_acc : {100*total_train_correct_cat3/train_data_len:.2f}% - ({total_train_correct_cat3}/{train_data_len}) \n")
    

    
    print("검증셋 시작")
    
    val_data_len = validation_dataset.__len__()
    model.eval()   
    model_preds = []
    true_labels = []    
    val_loss = []    
    total_val_correct_cat1 = 0
    total_val_correct_cat2 = 0
    total_val_correct_cat3 = 0
    with torch.no_grad():
        for i,data_batch in enumerate(validation_loader):
            print(f'검증중...{i}')
            if i==283:
                print('엣헴')
            img = data_batch['image']
            text = data_batch['text']
            cat1 = data_batch['label_1']
            cat2 = data_batch['label_2']
            cat3 = data_batch['label_3']
            mask = data_batch['mask']
            
            
            img = img.float().to(device)
            text = text.to(device)
            mask = mask.to(device)
            cat1 = cat1.to(device)
            cat2 = cat2.to(device)
            cat3 = cat3.to(device)
            
            cat1_p,cat2_p,cat3_p = model(img, text, mask, device)
            
            
            loss1 = criterion(cat1_p,cat1)
            loss2 = criterion(cat2_p,cat2)
            loss3 = criterion(cat3_p,cat3)
            
            total_loss = loss1+loss2+loss3
                
            _, predicted_1 = torch.max(cat1_p, 1) 
            correct_1 = (predicted_1 == cat1).sum().item()        
            total_val_correct_cat1 += correct_1        
            
            _, predicted_2 = torch.max(cat2_p, 1) 
            correct_2 = (predicted_2 == cat2).sum().item()        
            total_val_correct_cat2 += correct_2        
            
            _, predicted_3 = torch.max(cat3_p, 1) 
            correct_3 = (predicted_3 == cat3).sum().item()        
            total_val_correct_cat3 += correct_3        
                
            
            val_loss.append(total_loss.item())
            
            
        
        print(f"epoch {epoch} val end!!! val loss : {np.mean(val_loss):.3f} \t cat1_acc : {100*total_val_correct_cat1/val_data_len:.2f}% - ({total_val_correct_cat1}/{val_data_len}) cat2_acc : {100*total_val_correct_cat2/val_data_len:.2f}% - ({total_val_correct_cat2}/{val_data_len}) cat3_acc : {100*total_val_correct_cat3/val_data_len:.2f}% - ({total_val_correct_cat3}/{val_data_len}) \n\n")    
        log.info(f"epoch {epoch} val end!!! val loss : {np.mean(val_loss):.3f} \t cat1_acc : {100*total_val_correct_cat1/val_data_len:.2f}% - ({total_val_correct_cat1}/{val_data_len}) cat2_acc : {100*total_val_correct_cat2/val_data_len:.2f}% - ({total_val_correct_cat2}/{val_data_len}) cat3_acc : {100*total_val_correct_cat3/val_data_len:.2f}% - ({total_val_correct_cat3}/{val_data_len}) \n\n")
    
    writer.add_scalars("loss",{"tr_loss":tr_loss,"val loss":np.mean(val_loss)},epoch)
    writer.add_scalars("acc",{"tr_acc":total_train_correct_cat3/train_data_len,"val_acc":total_val_correct_cat3/val_data_len},epoch)
    torch.save(model.state_dict(),'./'+str(epoch)+".pth")
    

    
test_dataset = CustomDataset('./test.csv',tokenizer,test_transform,infer=True)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id,infer=True))
    
model.to(device)
model.eval()

model_preds = []
count = 0
with torch.no_grad():
    for data_batch in test_loader:
        count  += len(data_batch['image'])

        img = data_batch['image']
        text = data_batch['text']
        mask = data_batch['mask']

        img = img.float().to(device)
        text = text.to(device)
        mask = mask.to(device)
        
        cat1_p,cat2_p,model_pred = model(img, text,mask,device) 
        model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()

result = []
for i in model_preds:
    result.append(train_all_dataset.num2label_3_level[i])


submit = pd.read_csv('./sample_submission.csv')
submit['cat3'] = result

submit.to_csv('./submit.csv', index=False)
