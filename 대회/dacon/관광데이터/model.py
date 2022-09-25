import torch.nn as nn
import torch
from resnet import ResNet,block
class CustomModel(nn.Module):
    def __init__(self,vocab_size, num_classes):
        super(CustomModel, self).__init__()
        
        self.embedding_dim = 1024
        self.vocab_size = vocab_size
        
        # Image
        self.image_extract = ResNet(block, [3, 4, 6, 3],image_channels=3)       

        
        # Text
        self.bon_embed = nn.Embedding(vocab_size,self.embedding_dim,padding_idx=1)
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.nlp_bn = nn.BatchNorm1d(self.embedding_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(4160, num_classes)
        )
            

    def forward(self, img, text,text_len):
        img_feature = self.image_extract(img)
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

