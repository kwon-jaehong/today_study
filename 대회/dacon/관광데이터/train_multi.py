
from dataset import CustomDataset,MyCollate
from model import CustomModel
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
from kobert_tokenizer import KoBERTTokenizer
from torch.optim.lr_scheduler import LambdaLR
import torchinfo

from torch.utils.tensorboard import SummaryWriter


# writer = SummaryWriter('runs/experiment_1')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

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
        mask = data_batch['mask']
        
        
        img = img.float().to(rank)
        text = text.to(rank)
        label = label.to(rank)
        mask = mask.to(rank)
        optimizer.zero_grad()
        
        model_pred = model(img, text,mask,rank)        
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
            ## ??????????????? ???????????? ????????? ????????? ?????? ??????

        # print(lr)
            
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
    val_total_loss = []    
    total_val_correct = 0
    with torch.no_grad():
        for i,data_batch in enumerate(validation_loader):
            img = data_batch['image']
            text = data_batch['text']
            label = data_batch['label']
            mask = data_batch['mask']
        
            img = img.float().to(rank)
            text = text.to(rank)
            label = label.to(rank)
            mask = mask.to(rank)
            
            model_pred = model(img, text,mask,rank) 
            
            val_loss = criterion(model_pred, label)            
            _, predicted = torch.max(model_pred, 1) 
            val_correct = (predicted == label).sum().item()             
            val_correct = torch.tensor(val_correct,dtype=torch.long).to(rank)       
               
           
            dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            
                       
            total_val_correct += val_correct.detach().cpu()
            val_total_loss.append(val_loss.item())  

        

        if (rank==0):
            print(f"epoch {epoch} val end!!! val loss : {np.mean(val_total_loss)/args.world_size:.3f} \t acc : {100*total_val_correct/(val_data_len*args.world_size):.2f}% - ({total_val_correct}/{val_data_len*args.world_size}) \n\n")    

        return total_val_correct/(val_data_len*args.world_size),np.mean(val_total_loss)/args.world_size

def trainer(rank, world_size, args):
    setup(rank, world_size)
    # seed_everything(41) # Seed ??????
    

    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.world_size = world_size
    
    train_transform = A.Compose([
                            A.Resize(args.image_size,args.image_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    
    

    train_all_dataset = CustomDataset(args.train_path,tokenizer,train_transform)
    label_info = train_all_dataset.lable2num


    dataset_size = len(train_all_dataset)
    train_size = int(dataset_size * args.train_data_rate)
    validation_size = dataset_size - train_size



    ## ???????????? ?????????
    train_dataset, validation_dataset = random_split(train_all_dataset, [train_size, validation_size])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank)

    validation_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank)

    

    ## ????????? ??????
    pad_token_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.numworker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=train_sampler)
    validation_loader = DataLoader(validation_dataset, batch_size = args.batch_size, num_workers = args.numworker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=validation_sampler)
        
    
    model = CustomModel(tokenizer,len(label_info)).to(rank)
    

    
    model = DDP(model,device_ids=[rank],find_unused_parameters=True)
    
    
    
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
        
    ## ????????? ?????? ?????? ??????    
    t_total = len(train_loader) * args.epochs
    warmup_steps = 0
    # lr ????????? ??????????????? ????????????
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
    
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=48, metavar='N',
                        help='input batch size for training (default: 16)')
    
    parser.add_argument('--image_size', type=int, default=224, metavar='N',
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
      print("?????? GPU ?????? ??????", torch.cuda.device_count())
      
    mp.spawn(trainer, args=(world_size, args), nprocs=world_size, join=True)
    
    
if __name__ == '__main__':
    main()