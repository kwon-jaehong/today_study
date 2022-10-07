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
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
from torch.optim.lr_scheduler import LambdaLR
import logging as log
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, Sequence
from model import CustomModel
import pytorch_lightning as pl
os.chdir('../')
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':20,
    'LEARNING_RATE':2e-5,
    'BATCH_SIZE':6,
    'SEED':41,
    'TRAIN_RATE':0.9,
    'NUM_WORKERS':4
}



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
    
    

print("asd")
model = CustomModel(tokenizer,len(label_info))

# trainer = pl.Trainer(fast_dev_run=True)

# trainer.fit(model,train_dataloaders=train_loader)