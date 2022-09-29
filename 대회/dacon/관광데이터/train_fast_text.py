
import argparse
import random
import torch
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import cv2

from torch.utils.tensorboard import SummaryWriter

from konlpy.tag import Mecab

# 토치 텍스트 관련 참조자료
# https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def cleanup():
    dist.destroy_process_group()
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)




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
    


class FasttextmultimodalModel(nn.Module):
    def __init__(self,vocab_size, num_classes):
        super(FasttextmultimodalModel, self).__init__()
        
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



def trainer(rank, world_size, args):
    setup(rank, world_size)
    # seed_everything(41) # Seed 고정
    MAX_VOCAB_SIZE = 100000

    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.world_size = world_size
    
    train_transform = A.Compose([
                            A.Resize(args.image_size,args.image_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])
    


    train_all_dataset = CustomDataset(args.train_path,MAX_VOCAB_SIZE,train_transform)
    label_info = train_all_dataset.lable2num


    dataset_size = len(train_all_dataset)
    train_size = int(dataset_size * args.train_data_rate)
    validation_size = dataset_size - train_size



    ## 데이터셋 나누기
    train_dataset, validation_dataset = random_split(train_all_dataset, [train_size, validation_size])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank)

    validation_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank)

     
    ## 데이터 로더
    pad_token_id = train_all_dataset.TEXT.stoi['<pad>']
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.numworker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=train_sampler)
    validation_loader = DataLoader(validation_dataset, batch_size = args.batch_size, num_workers = args.numworker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=validation_sampler)
        
    
    model = FasttextmultimodalModel(len(train_all_dataset.TEXT.stoi),len(label_info)).to(rank)
    

    
    model = DDP(model,device_ids=[rank],find_unused_parameters=True)
    
    
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
        
    ## 학습할 최종 스텝 계산    
    t_total = len(train_loader) * args.epochs
    warmup_steps = 0
    # lr 조금씩 감소시키는 스케줄러
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    

    criterion = nn.CrossEntropyLoss().to(rank)
    
    if rank == 0:
        tb_write = SummaryWriter('runs/experiment_1')
        
    for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, epoch, rank,criterion,scheduler)
        lr,train_acc,train_loss = train(args, model, train_loader, optimizer, epoch, rank, criterion,scheduler)
        val_acc,val_loss= val(args, model, validation_loader, epoch, rank, criterion)
        if rank == 0:
            tb_write.add_scalars("acc",{"train_acc":train_acc,"val_acc":val_acc},epoch)
            tb_write.add_scalars("loss",{"train_loss":train_loss,"val_loss":val_loss},epoch)
            tb_write.add_scalar("lr",lr,epoch)        
    if rank == 0:
        tb_write.close()
    cleanup()
    

def train(args, model, train_loader, optimizer, epoch, rank,criterion,scheduler):
    # if rank == 0:
    #     tb_write = SummaryWriter('runs/experiment_1')
        
    model.train()
    train_loss = []   
    train_data_len = len(train_loader)*args.batch_size
    total_train_correct = 0
    
    for i,data_batch in enumerate(train_loader):
        img = data_batch['image']
        text = data_batch['text']
        label = data_batch['label']
        text_len = data_batch['text_len']
        
        img = img.float().to(rank)
        text = text.to(rank)
        label = label.to(rank)
        text_len = text_len.to(rank)
        optimizer.zero_grad()
        
        model_pred = model(img, text,text_len)    
        loss = criterion(model_pred, label)
        loss.backward()
        
        
        
        optimizer.step()                
        scheduler.step()
        _, predicted = torch.max(model_pred, 1) 
        correct = (predicted == label).sum().item()    
        correct = torch.tensor(correct,dtype=torch.long).to(rank)
            
               
        
        
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        
        total_train_correct += correct.detach().cpu()
        train_loss.append(loss.item())
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        if(rank==0):            
            print(f'epoch {epoch}/{args.epochs+1} {((i*args.batch_size)+len(label))*args.world_size}/{train_data_len*args.world_size} train loss {loss.item()/args.world_size:.4f} acc : {100*correct/(len(label)*args.world_size):.2f}% - ({correct}/{len(label)*args.world_size})')
            # print(lr)
            
    tr_loss = np.mean(train_loss) / args.world_size
    if(rank==0):
            
        print(f"\nepoch {epoch} train end!!! \t train batch loss : {tr_loss:.4f}\t total acc : {100*total_train_correct/(train_data_len*args.world_size):.2f}% - ({total_train_correct}/{train_data_len*args.world_size}) \n")
        
        # for param_group in optimizer.param_groups:
        #     lr = param_group['lr']
        # print(f"lr : {lr}")
        
        # tb_write.add_scalar("learing_rate",lr,epoch)
        # tb_write.add_scalar("train_acc",total_train_correct/(train_data_len*args.world_size),epoch)
        # tb_write.add_scalar("train_loss",tr_loss,epoch)
        # tb_write.close()
    return lr,total_train_correct/(train_data_len*args.world_size),tr_loss
        # torch.save(model.module.state_dict(), './model.pth')
        
def val(args, model, validation_loader, epoch, rank,criterion):
    # if rank == 0:
    #     tb_write = SummaryWriter('runs/experiment_1')
        
    val_data_len = len(validation_loader)*args.batch_size
    model.eval()    
    val_loss = []    
    total_val_correct = 0
    with torch.no_grad():
        for i,data_batch in enumerate(validation_loader):
            img = data_batch['image']
            text = data_batch['text']
            label = data_batch['label']
            text_len = data_batch['text_len']
        
            img = img.float().to(rank)
            text = text.to(rank)
            label = label.to(rank)
            text_len = text_len.to(rank)
            
            model_pred = model(img, text,text_len)    
            
            loss = criterion(model_pred, label)
            
            _, predicted = torch.max(model_pred, 1) 
            correct = (predicted == label).sum().item() 
            total_val_correct+=correct
            
            correct = torch.tensor(correct,dtype=torch.long).to(rank)    
     
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            val_loss.append(loss.item())  
            total_val_correct += correct.detach().cpu()


        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        if (rank==0):
            print(f"epoch {epoch} val end!!! val loss : {np.mean(val_loss)/args.world_size:.3f} \t acc : {100*total_val_correct/(val_data_len*args.world_size):.2f}% - ({total_val_correct}/{val_data_len*args.world_size}) \n\n")    
            # tb_write.add_scalar("val_acc",total_val_correct/(val_data_len*args.world_size),epoch)
            # tb_write.add_scalar("val_loss",np.mean(val_loss)/args.world_size,epoch)            
            # tb_write.close()
        return total_val_correct/(val_data_len*args.world_size),np.mean(val_loss)/args.world_size

    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                        help='input batch size for training (default: 16)')
    
    parser.add_argument('--image_size', type=int, default=128, metavar='N',
                        help='input image size for training (default: 224)')
    
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train ')
    
    parser.add_argument('--numworker', type=int, default=4, metavar='N',
                        help='worker')
    
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    parser.add_argument('--gpus', type=int, default=2, metavar='N',
                        help='Number of GPUs')
    
    parser.add_argument('--train_data_rate', type=float, default=0.9,
                        help='train / val split rate')
    

    parser.add_argument('--train_path', default="./train.csv",
                        help='train file path')
    
    parser.add_argument('--test_path', default="./test.csv",
                        help='test file path')
    
    parser.add_argument('--load_model_path', default='./1model.pth',
                        help='load_model_path')


    args = parser.parse_args()

    world_size = args.gpus

    if torch.cuda.device_count() > 1:
      print("병렬 GPU 사용 가능", torch.cuda.device_count())
      
    mp.spawn(trainer, args=(world_size, args), nprocs=world_size, join=True)
    
    
if __name__ == '__main__':
    main()
    





    


























