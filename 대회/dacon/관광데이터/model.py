import torch.nn as nn
import torch
from resnet import ResNet,block
from transformers import BertModel
import torch.nn.functional as F
class imageEmbeddings(nn.Module):
    def __init__(self):
        super(imageEmbeddings,self).__init__()        
        
        # Image 이미지 레즈넷으로 피쳐 뽑아오고 projection
        self.image_extract = ResNet(block, [3, 4, 6, 3],image_channels=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_embeddings = nn.Linear(2048,768) ## 이미지 2048 차원을 projection 시켜 버트 768 차원으로 만듬

    def forward(self,x):
        
        x = self.image_extract(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=2)
        x = x.transpose(1, 2).contiguous()
        # torch.Size([16, 1, 2048]) 토큰 3개 
        image_embedding = self.proj_embeddings(x)
        # torch.Size([16, 1, 768]) 768 차원으로 projcetion함으로써, 텍스트와 차원을 맞추어줌       

        return image_embedding


class CustomModel(nn.Module):
    def __init__(self,tokenizer,num_classes):
        super(CustomModel, self).__init__()
        self.dropout_p = 0.1
        self.hidden_dim = 768
        self.lstm_num_layers = 2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.num_classes = num_classes
        
        self.tokenizer = tokenizer
        
        # text 임베딩용 bert 모델
        self.transformer = BertModel.from_pretrained('skt/kobert-base-v1')
        ## kobert 파라미터 freeze
        for param in self.transformer.parameters():
            param.requires_grad_(False)
            

        self.lstm = nn.LSTM(self.hidden_dim,self.hidden_dim,self.lstm_num_layers,batch_first=True)
        self.text_bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.text_bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.text_bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.text_fc1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.text_fc2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.text_predict = nn.Linear(self.hidden_dim,num_classes)
        
        
        # Image
        self.image_encoder = imageEmbeddings()
        self.image_bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.image_bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.image_bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.image_fc1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.image_fc2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.image_predict = nn.Linear(self.hidden_dim,num_classes)

        
        
        
        ## fusion
        self.fusion_fc1 = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.fusion_fc2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.fusion_bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fusion_bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.fusion_predict = nn.Linear(self.hidden_dim,num_classes)
        
        
        
        # self.p_img_bn = nn.BatchNorm1d(self.num_classes)
        # self.p_text_bn = nn.BatchNorm1d(self.num_classes)
        # self.p_fusion_bn = nn.BatchNorm1d(self.num_classes)
        
        
        self.final_predict = nn.Linear(self.hidden_dim,num_classes)
        # self.final_bn = nn.BatchNorm1d(num_classes)



            

    def forward(self, img, text,mask ,device):


        img_embeddings = self.dropout(self.image_bn1(self.image_encoder(img).squeeze()))
        img_f = self.dropout(self.image_bn1(self.image_fc1(img_embeddings)))
        img_f = self.dropout(self.image_bn2(self.image_fc2(img_f)))
        
        # img_embeddings.shape = torch.Size([6, 1, 768])


        ## 텍스트 타입 토큰 선언 및 임베딩
        token_type_ids = torch.ones(text.size(), dtype=torch.long, device=device)   
        txt_embeddings = self.transformer.embeddings(input_ids=text,token_type_ids=token_type_ids)
        # txt_embeddings.shape =  torch.Size([6, 392, 768])
        
        h0 = torch.zeros(self.lstm_num_layers,text.shape[0],self.hidden_dim).to(device)
        c0 = torch.zeros(self.lstm_num_layers,text.shape[0],self.hidden_dim).to(device)
        
        lstm_output,_ = self.lstm(txt_embeddings,(h0,c0))
        text_f = self.text_fc1(self.dropout(self.text_bn1(lstm_output[:,-1,:])))
        text_f = self.text_fc2(self.dropout(self.text_bn2(text_f)))
        
        ##
        fusion_f = torch.cat([img_f, text_f], axis=1)
        # torch.Size([6, 1536])        
        fusion_f = self.dropout(self.fusion_bn1(self.fusion_fc1(fusion_f)))
        fusion_f = self.dropout(self.fusion_bn2(self.fusion_fc2(fusion_f)))
        
        # torch.Size([6, 768])
        

        
        # p_img = self.dropout(self.relu(self.p_img_bn(self.image_predict(img_f))))
        # p_text = self.dropout(self.relu(self.p_text_bn(self.text_predict(text_f))))
        # p_fusion = self.dropout(self.relu(self.p_fusion_bn(self.fusion_predict(fusion_f))))
        
        
        # fusion_p = torch.cat([p_img,p_text,p_fusion], axis=1)
        output = self.final_predict(fusion_f)
        
        
        
        return output

