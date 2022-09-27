
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


   
    

    
# test_dataset = CustomDataset('./test.csv',tokenizer,test_transform,infer=True)
# test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id,infer=True))
    
# model.to(device)
# model.eval()

# model_preds = []
# count = 0
# with torch.no_grad():
#     for data_batch in test_loader:
#         count  += len(data_batch['image'])

#         img = data_batch['image']
#         text = data_batch['text']
#         mask = data_batch['mask']

#         img = img.float().to(device)
#         text = text.to(device)
#         mask = mask.to(device)
        
#         model_pred = model(img, text,mask,device) 
#         model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()

# result = []
# for i in model_preds:
#     result.append(train_all_dataset.num2label[i])


# submit = pd.read_csv('./sample_submission.csv')
# submit['cat3'] = result

# submit.to_csv('./submit.csv', index=False)






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
    if rank == 0:
        tb_write = SummaryWriter('runs/experiment_1')
        
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
        
        if(rank==0):
            ## 러닝레이트 감소하고 있는거 보려면 주석 해제
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            # print(lr)
            
            print(f'epoch {epoch}/{args.epochs+1} {((i*args.batch_size)+len(label))*args.world_size}/{train_data_len*args.world_size} train loss {loss.item()/args.world_size:.4f} acc : {100*correct/(len(label)*args.world_size):.2f}% - ({correct}/{len(label)*args.world_size})')
            print(lr)
            
    tr_loss = np.mean(train_loss) / args.world_size
    if(rank==0):
            
        print(f"\nepoch {epoch} train end!!! \t train batch loss : {tr_loss:.4f}\t total acc : {100*total_train_correct/(train_data_len*args.world_size):.2f}% - ({total_train_correct}/{train_data_len*args.world_size}) \n")
        
        # for param_group in optimizer.param_groups:
        #     lr = param_group['lr']
        # print(f"lr : {lr}")
        
        tb_write.add_scalar("learing_rate",lr,epoch)
        tb_write.add_scalar("train_acc",total_train_correct/(train_data_len*args.world_size),epoch)
        tb_write.add_scalar("train_loss",tr_loss,epoch)
        tb_write.close()
        
        torch.save(model.module.state_dict(), './model.pth')
        
        
        
def val(args, model, validation_loader, epoch, rank,criterion):
    if rank == 0:
        tb_write = SummaryWriter('runs/experiment_1')
        
    val_data_len = len(validation_loader)*args.batch_size
    model.eval()    
    val_loss = []    
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
            
            loss = criterion(model_pred, label)
            
            _, predicted = torch.max(model_pred, 1) 
            correct = (predicted == label).sum().item() 
            total_val_correct+=correct
            
            correct = torch.tensor(correct,dtype=torch.long).to(rank)          
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            
            total_val_correct += correct.detach().cpu()
            val_loss.append(loss.item())

        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        if (rank==0):
            print(f"epoch {epoch} val end!!! val loss : {np.mean(val_loss)/args.world_size:.3f} \t acc : {100*total_val_correct/(val_data_len*args.world_size):.2f}% - ({total_val_correct}/{val_data_len*args.world_size}) \n\n")    
            tb_write.add_scalar("val_acc",total_val_correct/(val_data_len*args.world_size),epoch)
            tb_write.add_scalar("val_loss",np.mean(val_loss)/args.world_size,epoch)            
            tb_write.close()

def trainer(rank, world_size, args):
    setup(rank, world_size)
    # seed_everything(41) # Seed 고정
    

    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.world_size = world_size
    
    train_transform = A.Compose([
                            A.Resize(args.image_size,args.image_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])
    test_transform = A.Compose([
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
    pad_token_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.numworker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=train_sampler)
    validation_loader = DataLoader(validation_dataset, batch_size = args.batch_size, num_workers = args.numworker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=validation_sampler)
        
    
    model = CustomModel(tokenizer,len(label_info)).to(rank)
    
    args.load_model_path = "./temp.pth"
    if args.load_model_path != None:
        model.load_state_dict(torch.load(args.load_model_path))
    
    model = DDP(model,device_ids=[rank])
    
    
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        
    ## 학습할 최종 스텝 계산    
    t_total = len(train_loader) * args.epochs
    warmup_steps = 0
    # lr 조금씩 감소시키는 스케줄러
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    

    criterion = nn.CrossEntropyLoss().to(rank)
    
    for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, epoch, rank,criterion,scheduler)
        train(args, model, train_loader, optimizer, epoch, rank, criterion,scheduler)
        val(args, model, validation_loader, epoch, rank, criterion)
    
    cleanup()
    
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 16)')
    
    parser.add_argument('--image_size', type=int, default=224, metavar='N',
                        help='input image size for training (default: 224)')
    
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train ')
    
    parser.add_argument('--numworker', type=int, default=4, metavar='N',
                        help='worker')
    
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
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