import random
import torch
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset import CustomDataset,MyCollate
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from typing import Optional, Sequence
from resnet import ResNet,block
from transformers import BertModel
import pytorch_lightning as pl

class imageEmbeddings(nn.Module):
    def __init__(self,tokenizer,embeddings):
        super(imageEmbeddings,self).__init__()
        
        self.tokenizer = tokenizer
        
        ## 버트 임베딩 정보 불러옴
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=0.1)
        
        
        
        # Image 이미지 레즈넷으로 피쳐 뽑아오고 projection
        self.image_extract = ResNet(block, [3, 8, 36, 3],image_channels=3)
        self.avgpool = nn.AdaptiveAvgPool2d((16, 1))
        self.proj_embeddings = nn.Linear(2048,768) ## 이미지 2048 차원을 projection 시켜 버트 768 차원으로 만듬
        
              
        self.start_token = tokenizer.cls_token_id
        self.end_token = tokenizer.sep_token_id
        
    def forward(self,x,device):
        
        x = self.image_extract(x)
        
        # torch.Size([6, 2048, 7, 7])
        x = self.avgpool(x)
        # torch.Size([6, 2048, 3, 1])
        
        x = torch.flatten(x, start_dim=2)
        x = x.transpose(1, 2).contiguous()
        # torch.Size([16, 3, 2048]) 토큰 3개 
        token_embeddings = self.proj_embeddings(x)
        # torch.Size([16, 3, 768]) 768 차원으로 projcetion 이로써, 텍스트와 차원을 맞추어줌
        

                
        seq_length = token_embeddings.size(1)        
        batch_size = x.shape[0]
        # 이미지 앞에 스타트 토큰 부여 (CLS)
        start_token_embeds = self.word_embeddings(torch.tensor(self.start_token).expand(batch_size).to(device))
        seq_length += 1
        token_embeddings = torch.cat([start_token_embeds.unsqueeze(1), token_embeddings], dim=1)

        ## 이미지 뒤에 end 토큰 부여 (SEP)
        end_token_embeds = self.word_embeddings(torch.tensor(self.end_token).expand(batch_size).to(device))
        seq_length += 1
        token_embeddings = torch.cat([token_embeddings, end_token_embeds.unsqueeze(1)], dim=1)

        ## 포지션 임베딩 값
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        ## 이미지타입 부여 (segmention)
        token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
        
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        

        return embeddings

class CustomModel(pl.LightningModule):
    def __init__(self,tokenizer,num_cat3_classes,drop_p=0.3):
        super(CustomModel, self).__init__()
        
        self.loss_funtion = FocalLoss()
        
        self.tokenizer = tokenizer
        
        # text 임베딩용 bert 모델
        self.transformer = BertModel.from_pretrained('skt/kobert-base-v1')
                
        # Image
        self.image_encoder = imageEmbeddings(tokenizer,self.transformer.embeddings)
        

        self.dropout = nn.Dropout(drop_p)
        self.classifier = nn.Linear(768, num_cat3_classes)

    
    

    def forward(self, img, text,mask ,device):

        img_token_embeddings = self.image_encoder(img,device)
        
        input_image_shape = img_token_embeddings.size()[:-1]

        ## 텍스트 타입 토큰 선언 및 임베딩
        token_type_ids = torch.ones(text.size(), dtype=torch.long, device=device)   
        txt_embeddings = self.transformer.embeddings(input_ids=text,token_type_ids=token_type_ids)
        
        ## 둘 정보를 concat       
        embedding_output = torch.cat([img_token_embeddings, txt_embeddings], axis=1)
        input_shape = embedding_output.size()[:-1]

        
        attention_mask = torch.cat([torch.ones(input_image_shape, device=device, dtype=torch.long), mask], dim=1)
        encoder_attention_mask = torch.ones(input_shape, device=device)
        
        extended_attention_mask = self.get_extended_mask(attention_mask)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        head_mask = [None] * 12
        
        
        
        encoder_outputs = self.transformer.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=True,
        )
        
        sequence_output = encoder_outputs[0]
        # sequence_output = torch.Size([4, 512, 768])
        
        pooled_output = self.transformer.pooler(sequence_output)
       
        pooled_output = self.dropout(pooled_output)       
        
        logits = self.classifier(pooled_output) 
        return logits

    def get_extended_mask(self,attention_mask):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        return extended_attention_mask
    
    def invert_attention_mask(self,encoder_attention_mask):
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(torch.float32)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(torch.float32).min

        return encoder_extended_attention_mask
    def training_step(self, batch,batch_idx):
        x,y = self(batch)
        y_hat = self(x)
        loss = self.loss_funtion(x,y)
        
        # loss.backw()
        
        return {'loss':loss}
    
    def train_dataloader(self):
        return 0
    
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=0.00003, eps=1e-08)
        
        
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
    
    
if __name__ == "__main__":
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