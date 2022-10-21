import torch.nn as nn
import torch
from transformers import AutoModel
from torchvision.models import resnext101_64x4d
from transformers import BeitFeatureExtractor, BeitModel

class imageEmbeddings(nn.Module):
    def __init__(self,tokenizer,transformer,image_encoder_dropout_p,image_token_size):
        super(imageEmbeddings,self).__init__()
        
        self.tokenizer = tokenizer
        
        self.embedding_dim = transformer.config.hidden_size
        # (tokenizer,self.transformer.embeddings,image_encoder_dropout_p,image_token_size,resnet_choice)
        
        self.drop_out_p = image_encoder_dropout_p
        ## 부모 메인 모델의 임베딩 정보 불러옴
        self.position_embeddings = transformer.embeddings.position_embeddings
        self.token_type_embeddings = transformer.embeddings.token_type_embeddings
        self.word_embeddings = transformer.embeddings.word_embeddings
        self.LayerNorm = transformer.embeddings.LayerNorm
        self.dropout = nn.Dropout(p=self.drop_out_p)
        
        

        
        # Image 이미지를 beit모델로 token화

        self.image_extract = BeitModel.from_pretrained('microsoft/beit-base-patch16-224',cache_dir="./temp")
        # self.image_extract = BeitModel.from_pretrained('microsoft/beit-large-patch16-512',cache_dir="./temp")
        
        ## 이미지 인코더 파라미터 프리징
        for para in self.image_extract.parameters():
            para.requires_grad = False
               
        ## 토큰화된 이미지 평균 풀링 진행
        self.avgpool = nn.AdaptiveAvgPool2d((image_token_size, self.embedding_dim))
        
        ## 임베딩된 이미지 한번 -> prjetion
        self.proj_embeddings = nn.Linear(self.embedding_dim,self.embedding_dim) 
              
        self.start_token = tokenizer.cls_token_id
        self.end_token = tokenizer.sep_token_id
        
    def forward(self,x,device):
                
        x = self.image_extract(x)
        
        ## beti로 임베딩된 이미지 상태값들을 평균 풀링, 모양 조정 
        x = self.avgpool(x[0])
        
        ## projection
        token_embeddings = self.proj_embeddings(x)
                        
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

class CustomModel(nn.Module):
    def __init__(self,tokenizer,num_cat3_classes,main_model_dropout_p,image_encoder_dropout_p,image_token_size):
        super(CustomModel, self).__init__()
        self.tokenizer = tokenizer
        
        # text 임베딩용 bert 모델
        self.transformer = AutoModel.from_pretrained('klue/roberta-base',cache_dir="./temp")     
        # self.transformer = AutoModel.from_pretrained('klue/roberta-large',cache_dir="./temp")         
        self.main_embedding_size = self.transformer.config.hidden_size
               
               
        # Image 토큰화
        self.image_encoder = imageEmbeddings(tokenizer,self.transformer,image_encoder_dropout_p,image_token_size)
        
        self.dropout = nn.Dropout(main_model_dropout_p)
        
        self.classifier = nn.Linear(self.main_embedding_size, num_cat3_classes)

            

    def forward(self, img, text,mask ,device):

        img_token_embeddings = self.image_encoder(img,device)
        
        input_image_shape = img_token_embeddings.size()[:-1]

        ## 텍스트 타입 토큰 선언 및 임베딩
        token_type_ids = torch.zeros(text.size(), dtype=torch.long, device=device)   
        
        ## robrat 텍스트 임베딩
        txt_embeddings = self.transformer.embeddings(input_ids=text,token_type_ids=token_type_ids)
      
        ## 둘 토큰 정보를 concat       
        embedding_output = torch.cat([img_token_embeddings, txt_embeddings], axis=1)
        input_shape = embedding_output.size()[:-1]
        
        ## 마스크 생성        
        attention_mask = torch.cat([torch.ones(input_image_shape, device=device, dtype=torch.long), mask], dim=1)
        encoder_attention_mask = torch.ones(input_shape, device=device)
        
        extended_attention_mask = self.get_extended_mask(attention_mask)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        head_mask = [None] * 24
        
        ## 이미지,텍스트 concat 토큰 정보를 인코딩
        encoder_outputs = self.transformer.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=True,
        )
        
        sequence_output = encoder_outputs[0]
        
        
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