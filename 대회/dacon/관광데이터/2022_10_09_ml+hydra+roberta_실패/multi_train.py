from d_set import CustomDataset,MyCollate
from model import CustomModel
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
from transformers import  AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
from torch.nn import functional as F
import pickle
import mlflow
from omegaconf import DictConfig
import hydra


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

def gather(tensor, tensor_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
  
    rank = dist.get_rank()
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert(tensor_list is not None)
        dist.gather(tensor, gather_list=tensor_list, group=group)
    else:
        dist.gather(tensor, dst=root, group=group)

def score_function(real, pred):
    return f1_score(real, pred, average="weighted")

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight 

    def forward(self, input, target):
        if len(target.shape) ==0:
            target = target.view(-1)
            
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return focal_loss
    
def train(args, model, train_loader, optimizer, epoch, rank,criterion,scheduler):        
    model.train()
    train_loss_list = []   
    train_data_len = len(train_loader)*args.batch_size
    total_train_correct = 0
    
    
    
    for i,t_data_batch in enumerate(train_loader):
        train_img = t_data_batch['image']
        train_text = t_data_batch['text']
        train_label = t_data_batch['label_3']
        train_mask = t_data_batch['mask']
        
        
        train_img = train_img.float().to(rank)
        train_text = train_text.to(rank)
        train_label = train_label.to(rank)
        train_mask = train_mask.to(rank)
        optimizer.zero_grad()
        
        model_pred = model(train_img, train_text,train_mask,rank)        
        train_loss = criterion(model_pred, train_label)
        train_loss.backward()
        
        
        optimizer.step()                
        scheduler.step()
        _, train_predicted = torch.max(model_pred, 1) 
        train_correct = (train_predicted == train_label).sum().item()    
        train_correct = torch.tensor(train_correct,dtype=torch.long).to(rank)
        
        
        dist.all_reduce(train_correct, op=dist.ReduceOp.SUM)
        # print("\n 각각 loss",rank," loss : ",loss.item(),"\n")
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        # print("\n 합친후 ",rank," loss : ",loss,"\n")
        
        # val이랑 train 변수 바꾸자
        
        total_train_correct += train_correct.detach().cpu()
        train_loss_list.append(train_loss.item())
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        if(rank==0):

        # print(lr)
            
            print(f'epoch {epoch}/{args.epochs} {((i*args.batch_size)+len(train_label.view(-1)))*args.gpus}/{train_data_len*args.gpus} train loss {train_loss.item()/args.gpus:.4f} acc : {100*train_correct/(len(train_label.view(-1))*args.gpus):.2f}% - ({train_correct}/{len(train_label.view(-1))*args.gpus})')
            # print(lr)
            
    tr_loss = np.mean(train_loss_list) / args.gpus
    if(rank==0):
        print(f"\nepoch {epoch} train end!!! \t train batch loss : {tr_loss:.4f}\t total acc : {100*total_train_correct/(train_data_len*args.gpus):.2f}% - ({total_train_correct}/{train_data_len*args.gpus}) \n")
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        # print(f"lr : {lr}")
        
    return lr,total_train_correct/(train_data_len*args.gpus),tr_loss   
        
def val(args, model, validation_loader, epoch, rank,criterion):        
    val_data_len = len(validation_loader)*args.batch_size
    model.eval()    
    val_total_loss = []    
    total_val_correct = 0
    
    if rank ==0:
        rank_root_preds = None
        rank_root_true = None
    
    val_model_preds = []
    val_true_labels = []
    with torch.no_grad():
        for val_data_batch in validation_loader:
            val_img = val_data_batch['image']
            val_text = val_data_batch['text']
            val_label = val_data_batch['label_3']
            val_mask = val_data_batch['mask']
        
            val_img = val_img.float().to(rank)
            val_text = val_text.to(rank)
            val_label = val_label.to(rank)
            val_mask = val_mask.to(rank)
            
            model_pred = model(val_img, val_text,val_mask,rank)          
            
            val_loss = criterion(model_pred, val_label)            
            _, val_predicted = torch.max(model_pred, 1) 
            val_correct = (val_predicted == val_label).sum().item()             
            val_correct = torch.tensor(val_correct,dtype=torch.long).to(rank)       
                          
            dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)

            total_val_correct += val_correct.detach().cpu()
            val_total_loss.append(val_loss.item())  
            
            
            # f1 웨이트 스코어 구하기 위함
            val_model_preds = model_pred.argmax(1).detach()
            val_true_labels = val_label.detach()

            prediction_list = [torch.zeros_like(val_model_preds) for _ in range(args.gpus)]
            true_list = [torch.zeros_like(val_true_labels) for _ in range(args.gpus)]
         
            if dist.get_rank() == 0:
                gather(val_model_preds, prediction_list)
                gather(val_true_labels, true_list)
                ## 첫번째 인자는 저장할 정보,두번째 인자는 담을 그릇
            else:
                gather(val_model_preds)
                gather(val_true_labels)
                
            if rank == 0:
                for i in range(0,args.gpus):
                    if rank_root_preds == None and rank_root_true== None:
                        rank_root_preds = prediction_list[i]
                        rank_root_true = true_list[i]
                    else:
                        rank_root_preds = torch.concat([rank_root_preds.view(-1),prediction_list[i].view(-1)])
                        rank_root_true = torch.concat([rank_root_true.view(-1),true_list[i].view(-1)])


        weighted_f1 = 0
        if (rank==0):
            weighted_f1 = score_function(rank_root_true.detach().cpu().numpy().tolist(), rank_root_preds.detach().cpu().numpy().tolist())

            print(f"epoch {epoch} val end!!! val loss : {np.mean(val_total_loss)/args.gpus:.3f} \t f1 score : {weighted_f1:.2f} \t acc : {100*total_val_correct/(val_data_len*args.gpus):.2f}% - ({total_val_correct}/{val_data_len*args.gpus}) \n\n")    
        return total_val_correct/(val_data_len*args.gpus),np.mean(val_total_loss)/args.gpus,weighted_f1
    

def trainer(rank, gpus, args):

    setup(rank, gpus)
    seed_everything(41) # Seed 고정
    
    if rank ==0:
        mlflow.log_params(dict(args))   

    

    train_transform = A.Compose([
                            A.Resize(args.image_size,args.image_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large",cache_dir="./temp")
    

    train_all_dataset = CustomDataset(args.train_path,tokenizer,train_transform)
    label_info = train_all_dataset.label_3_level_2num



    dataset_size = len(train_all_dataset)
    train_size = int(dataset_size * args.train_data_rate)
    validation_size = dataset_size - train_size
    ## 데이터셋 나누기
    train_dataset, validation_dataset = random_split(train_all_dataset, [train_size, validation_size])
    
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.gpus,
        rank=rank)

    validation_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset,
        num_replicas=args.gpus,
        rank=rank)

    ## 데이터 로더
    pad_token_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_worker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=train_sampler)
    validation_loader = DataLoader(validation_dataset, batch_size = args.batch_size, num_workers = args.num_worker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=validation_sampler)
        
    # (self,tokenizer,num_cat3_classes,main_model_dropout_p,image_encoder_dropout_p,image_token_size,resnet_choice):
    model = CustomModel(tokenizer,len(label_info),args.main_model_drop_out_p,args.image_encoder_drop_out_p,args.image_token_size,args.resnet_choice).to(rank)
    
    model = DDP(model,device_ids=[rank],find_unused_parameters=True)    
    
    no_decay = ["bias", "LayerNorm.weight"]
    for pa_name in  model.named_parameters():
        # if pa_name[0] not any(nd in n for nd in no_decay)
        print(pa_name[0])
    
    # not any = 256
    # any = 298
    # 학습 전체는 길이 554
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    model.parameters()
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        
    ## 학습할 최종 스텝 계산    
    t_total = len(train_loader) * args.epochs
    warmup_steps = 0
    # lr 조금씩 감소시키는 스케줄러
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    criterion = FocalLoss().to(rank)
    
    best_f1_score = 0
    for epoch in range(0, args.epochs):
        lr,train_acc,train_loss = train(args, model, train_loader, optimizer, epoch, rank, criterion,scheduler)
        val_acc,val_loss,weighted_f1= val(args, model, validation_loader, epoch, rank, criterion)
        
        if rank == 0:
            mlflow.log_metric("lr",lr,epoch)
            mlflow.log_metric("train_acc",train_acc,epoch)
            mlflow.log_metric("train_loss",train_loss,epoch)
            mlflow.log_metric("val_acc",val_acc,epoch)
            mlflow.log_metric("val_loss",val_loss,epoch)
            mlflow.log_metric("f1_score",weighted_f1,epoch)
            
            if best_f1_score < weighted_f1:
                best_f1_score = weighted_f1
    
                save_dict = {"epoch":epoch,"best_f1_score":best_f1_score}
                with open("./temp.pickle",'wb') as fw:
                    pickle.dump(save_dict,fw)
    

    cleanup()
    

@hydra.main(version_base=None, config_path='conf',config_name="config")
def main(config: DictConfig):
    if torch.cuda.device_count() > 1:
      print("병렬 GPU 사용 가능", torch.cuda.device_count())
      
    mp.spawn(trainer, args=(config.gpus, config), nprocs=config.gpus, join=True)
    

    with open('./temp.pickle','rb') as fr:
        best_f1_score = pickle.load(fr)
    
    return best_f1_score['best_f1_score']
    
if __name__ == '__main__':
    main()