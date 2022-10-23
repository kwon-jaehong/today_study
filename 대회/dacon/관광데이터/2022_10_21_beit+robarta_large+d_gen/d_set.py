import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import torch.nn as nn
import re
import os

    
class CustomDataset(Dataset):
    def __init__(self,df,csv_path,tokenizer,transforms,label_2_num,infer=False):
        self.tokenizer = tokenizer        
        self.df = df             
        self.label_2_num = label_2_num
        
        # 텍스트속 html 문법 제거
        self.text_list = list(self.df['overview'])
        for i,text in enumerate(self.text_list):
            self.text_list[i] = self.cleanhtml(text)
        
        data_dir_name,_ = os.path.split(csv_path)
        self.img_path_list = [os.path.join(data_dir_name,pt[2:]) for pt in list(self.df['img_path'])]        
        
        if infer==False:
            self.label_level_list_3 = list(self.df['cat3'])      
         

        self.transforms = transforms
        self.infer = infer
        
    def __getitem__(self, index):

        text = self.text_list[index]
        text_token = self.tokenizer.encode(text)

        ## 시작과 끝 토큰 떄기
        text_token = text_token[1:-1]
        # [2, 993, 6516, 5446, 5468, 2640, 6573, 6516, 4258, 7382, 6896, 3563, 7788, 3860, ...]
        # list(self.tokenizer.get_vocab().keys())[932:935]
        
        text_len = len(text_token)
        if text_len > 512: # 버트모델 maxlen 512 - 이미지 임베딩 토큰 3 - cls,sep 토큰 2 
            text_token = text_token[:512]
        
        ## Image
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']        
        # Label
        if self.infer:
            return image, torch.Tensor(text_token).view(-1).to(torch.long)
        else:
            label_3_level = self.label_2_num[self.label_level_list_3[index]]
            return image, torch.Tensor(text_token).view(-1).to(torch.long), torch.tensor([label_3_level],dtype=torch.long),index

    def __len__(self):
        return len(self.df)
    
    def cleanhtml(self,raw_html):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext


class MyCollate:
    def __init__(self, pad_idx,infer=False):
        ## pad idx 가져옴
        self.pad_idx = pad_idx
        self.infer = infer

    def __call__(self, batch):
        lens = [len(row[1]) for row in batch]
        bsz, max_seq_len = len(batch), max(lens)
        
        mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
        for i_batch, length in enumerate(lens):
            mask_tensor[i_batch, :length] = 1
        
        
        image = [item[0] for item in batch]  
        text = [item[1] for item in batch]  
        
        text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value = self.pad_idx) 
        
        if self.infer==False:
            # label_level_1 = [item[2] for item in batch]
            index_list = [item[3] for item in batch]
            label_level_3 = [item[2] for item in batch]  
            return {"image":torch.stack(image),"text":text,"mask":mask_tensor,"label_3":torch.stack(label_level_3).squeeze(),"index_list":index_list}
        else:
            return {"image":torch.stack(image),"text":text,"mask":mask_tensor}
    
            
            
            