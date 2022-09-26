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
writer = SummaryWriter('runs/experiment_1')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':16,
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

## 데이터셋 나누기
train_dataset, validation_dataset = random_split(train_all_dataset, [train_size, validation_size])

## 데이터 로더
pad_idx = train_all_dataset.TEXT.stoi['<pad>']
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_idx))
validation_loader = DataLoader(validation_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_idx))
    
    
    
model = CustomModel(vocab_size,len(label_info))
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])



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
        
        print(f"epoch {epoch} val end!!! val loss : {np.mean(val_loss):.3f} \t acc : {100*total_val_correct/val_data_len:.2f}% - ({total_val_correct}/{val_data_len}) \n\n")    

    
    writer.add_scalars("loss",{"tr_loss":tr_loss,"val loss":np.mean(val_loss)},epoch)
    writer.add_scalars("acc",{"tr_acc":total_train_correct/train_data_len,"val_acc":total_val_correct/val_data_len},epoch)
    # torch.save(model.state_dict(),'./'+str(epoch)+".pth")
    

test_dataset = CustomDataset('./test.csv',CFG['MAX_VOCAB_SIZE'],train_all_dataset.TEXT,test_transform,infer=True)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_idx))
    
model.to(device)
model.eval()

model_preds = []
count = 0
with torch.no_grad():
    for data_batch in test_loader:
        count  += len(data_batch['image'])

        img = data_batch['image']
        text = data_batch['text']
        text_len = data_batch['text_len']

        img = img.float().to(device)
        text = text.to(device)
        text_len = text_len.to(device)
        
        model_pred = model(img, text,text_len)
        model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()

result = []
for i in model_preds:
    result.append(train_all_dataset.num2label[i])


submit = pd.read_csv('./sample_submission.csv')
submit['cat3'] = result

submit.to_csv('./submit.csv', index=False)
