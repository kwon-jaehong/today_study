import pandas as pd
from d_set import CustomDataset,MyCollate
from model import CustomModel
import random
import torch
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import  AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
from torch.nn import functional as F
import pickle
from omegaconf import DictConfig
import hydra
from transformers import AutoTokenizer
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf,open_dict
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging
import shutil
import pathlib
from eda import custom_oversampling


logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

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
    
def train(args, model, train_loader, optimizer, epoch, rank,criterion,scheduler,k_n):        
    model.train()
    train_loss_list = []   
    train_data_len = len(train_loader)*args.parameters_.batch_size
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
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        
        optimizer.step()                
        scheduler.step()
        _, train_predicted = torch.max(model_pred, 1) 
        train_correct = (train_predicted == train_label).sum().item()    
        train_correct = torch.tensor(train_correct,dtype=torch.long).to(rank)
        
        
        dist.all_reduce(train_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)

                
        total_train_correct += train_correct.detach().cpu()
        train_loss_list.append(train_loss.item())
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            
        if(rank==0):            
            print(f'{k_n}_fold {epoch}/{args.env_.epochs-1} epoch {((i*args.parameters_.batch_size)+len(train_label.view(-1)))*args.env_.gpus}/{train_data_len*args.env_.gpus} train loss {train_loss.item()/args.env_.gpus:.4f} acc : {100*train_correct/(len(train_label.view(-1))*args.env_.gpus):.2f}% - ({train_correct}/{len(train_label.view(-1))*args.env_.gpus})')
   
            
    tr_loss = np.mean(train_loss_list) / args.env_.gpus
    if(rank==0):
        print(f"\n{k_n}_fold {epoch} epoch train end!!! \t train batch loss : {tr_loss:.4f}\t total acc : {100*total_train_correct/(train_data_len*args.env_.gpus):.2f}% - ({total_train_correct}/{train_data_len*args.env_.gpus}) \n")
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        
        # torch.save(model.module.state_dict(), './'+str(epoch)+'.pth')
        
    return lr,total_train_correct/(train_data_len*args.env_.gpus),tr_loss   
        
def val(args, model, validation_loader, epoch, rank,criterion,k_n):        
    val_data_len = len(validation_loader)*args.parameters_.batch_size
    model.eval()    
    val_total_loss = []    
    total_val_correct = 0
    
    if rank ==0:
        rank_root_preds = None
        rank_root_true = None
    
    val_model_preds = []
    val_true_labels = []
    val_pred_info_list =[]
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
            

            ## 검증 통계용으로 index list 저장
            ## df 인덱스, 예측 logit max index값, 타겟, 실제 로짓값
            pred_info =[ [index,np.argmax(pred_logit),target,pred_logit] for index,pred_logit,target in zip(val_data_batch['index_list'],model_pred.detach().cpu().tolist(),val_label.view(-1).detach().cpu().tolist())]
            val_pred_info_list = val_pred_info_list + pred_info
            
            
            
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

            prediction_list = [torch.zeros_like(val_model_preds) for _ in range(args.env_.gpus)]
            true_list = [torch.zeros_like(val_true_labels) for _ in range(args.env_.gpus)]
         
            if dist.get_rank() == 0:
                gather(val_model_preds, prediction_list)
                gather(val_true_labels, true_list)
                ## 첫번째 인자는 저장할 정보,두번째 인자는 담을 그릇
            else:
                gather(val_model_preds)
                gather(val_true_labels)
                
            if rank == 0:
                for i in range(0,args.env_.gpus):
                    if rank_root_preds == None and rank_root_true== None:
                        rank_root_preds = prediction_list[i]
                        rank_root_true = true_list[i]
                    else:
                        rank_root_preds = torch.concat([rank_root_preds.view(-1),prediction_list[i].view(-1)])
                        rank_root_true = torch.concat([rank_root_true.view(-1),true_list[i].view(-1)])


        weighted_f1 = 0
        if (rank==0):
            weighted_f1 = score_function(rank_root_true.detach().cpu().numpy().tolist(), rank_root_preds.detach().cpu().numpy().tolist())

            print(f"{k_n}_fold {epoch} val end!!! val loss : {np.mean(val_total_loss)/args.env_.gpus:.3f} \t f1 score : {weighted_f1:.2f} \t acc : {100*total_val_correct/(val_data_len*args.env_.gpus):.2f}% - ({total_val_correct}/{val_data_len*args.env_.gpus}) \n\n")    
        return total_val_correct/(val_data_len*args.env_.gpus),np.mean(val_total_loss)/args.env_.gpus,weighted_f1,val_pred_info_list
    

def trainer(rank, gpus, args):

    setup(rank, gpus)
    seed_everything(41) # Seed 고정
    


    if rank ==0:
        mlflow_client = MlflowClient()
        train_acc_list = [ [] for x in range(args.env_.epochs) ]
        train_loss_list = [ [] for x in range(args.env_.epochs) ]
        val_acc_list = [ [] for x in range(args.env_.epochs) ]
        val_loss_list = [ [] for x in range(args.env_.epochs) ]
        f1_socre_list = [ [] for x in range(args.env_.epochs) ]
        mlflow_run = mlflow_client.create_run(experiment_id=str(args.env_.experiments_id),tags={MLFLOW_RUN_NAME:args.env_.job_name})
        mlflow_run_id = mlflow_run.info.run_id   
        mlflow_client.log_dict(mlflow_run_id,dict(args.env_),"env.yaml")
        for key,value in dict(args.parameters_).items():
            mlflow_client.log_param(mlflow_run_id,key,value)
        
        


    train_transform = A.Compose([
                            A.Resize(args.env_.image_size,args.env_.image_size),
                            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base",cache_dir="./temp")
    # tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large",cache_dir="./temp")
    
    ## 라벨링 cat3를 dict로 작업
    df = pd.read_csv(args.env_.train_path,index_col=0)
    df = df.reset_index()
    set_level_list_3 = list(set(list(df['cat3'])))
    set_level_list_3.sort()            
    label_info= {label:i for i,label in enumerate(set_level_list_3)}


    df['kfold'] = -1
    if args.env_.k_fold_n==1:
        dataset_size = len(df)
        train_size = int(dataset_size * args.env_.train_data_rate)
        validation_size = dataset_size - train_size
        ## 데이터셋 나누기
        train_index_list, validation_index_list = random_split(df, [train_size, validation_size])   
        df.loc[train_index_list.indices,'kfold'] = 1
        df.loc[validation_index_list.indices,'kfold'] = 0
        
    else:
        ## df에 k폴드 번호 지정
        # folds = StratifiedKFold(n_splits=args.env_.k_fold_n, random_state=42, shuffle=True)
        folds = StratifiedKFold(n_splits=args.env_.k_fold_n)
        for i in range(args.env_.k_fold_n):
            df_idx, valid_idx = list(folds.split(df.values, df['cat3']))[i]
            valid = df.iloc[valid_idx]
            df.loc[df[df.id.isin(valid.id) == True].index.to_list(), 'kfold'] = i


    

    f_best_f1 = {}
    f_best_f1_valloss = {}
    for k_n in range(0,args.env_.k_fold_n):
    
        ## train, val k폴드로 df 불러옴
        train_df = df[df['kfold']!=k_n]
        val_df = df[df['kfold']==k_n]
        
        ## 데이터셋 증강 시킬것인지
        if args.parameters_.argument_text_data:
            train_df = custom_oversampling(train_df,'cat3',args.parameters_.argument_text_rank)

        if rank==0:
            train_df.to_csv(os.path.join(args.env_.data_save_dir,str(k_n)+'_flod_train_df.csv'))
            val_df.to_csv(os.path.join(args.env_.data_save_dir,str(k_n)+'_flod_val_df.csv'))
            mlflow_client.log_artifact(mlflow_run_id,os.path.join(args.env_.data_save_dir,str(k_n)+'_flod_train_df.csv'),'train_df')
        
                
            
        
        train_dataset = CustomDataset(train_df,args.env_.train_path,tokenizer,train_transform,label_info)
        validation_dataset = CustomDataset(val_df,args.env_.train_path,tokenizer,train_transform,label_info)

        ## 샘플러 셔플 -> 데이터 증강 했을때만
        if args.parameters_.argument_text_data:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.env_.gpus,rank=rank,shuffle=True)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.env_.gpus,rank=rank)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset,num_replicas=args.env_.gpus,rank=rank)



        ## 데이터 로더
        pad_token_id = tokenizer.pad_token_id

        train_loader = DataLoader(train_dataset, batch_size = args.parameters_.batch_size, num_workers = args.env_.num_worker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=train_sampler)
        validation_loader = DataLoader(validation_dataset, batch_size = args.parameters_.batch_size, num_workers = args.env_.num_worker,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id),sampler=validation_sampler)
            
            

        model = CustomModel(tokenizer,len(label_info),args.parameters_.main_model_drop_out_p,args.parameters_.image_encoder_drop_out_p,args.parameters_.image_token_size).to(rank)
        
        model = DDP(model,device_ids=[rank],find_unused_parameters=True)    
        
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.parameters_.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.parameters_.lr, eps=1e-8)
            
        ## 학습할 최종 스텝 계산    
        t_total = len(train_loader) * args.env_.epochs
        warmup_steps = 0
        # lr 조금씩 감소시키는 스케줄러
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        criterion = FocalLoss().to(rank)
        
        best_f1 = 0
        best_val_loss = 0
  
        
        for epoch in range(0, args.env_.epochs):
            
            lr,train_acc,train_loss = train(args, model, train_loader, optimizer, epoch, rank, criterion,scheduler,k_n)
            torch.save(model.module.state_dict(), os.path.join('fold_'+str(k_n)+'_epoch_'+str(epoch)+'_model.pth'))
            val_acc,val_loss,weighted_f1,val_pred_info_list= val(args, model, validation_loader, epoch, rank, criterion,k_n)
            
            
            ## 에폭별 val 예측결과값  자료 분석
            epoch_val_df = val_df.copy()
            epoch_val_df = epoch_val_df.reset_index(drop=True)
            epoch_val_df['predict_label_number'] = -1
            epoch_val_df['target'] = -1
            epoch_val_df['predict_correct'] = False
            epoch_val_df['predict_logit'] = pd.NaT         
            epoch_val_df['predict_logit'] = epoch_val_df['predict_logit'].astype(object)           
            ## val 결과값들 정제 하고 gpu rank별 csv로 만들어 저장
            for data in val_pred_info_list:
                index_number = data[0]
                predict_label_number = data[1]
                target_label_number = data[2]
                predict_logit = data[3]             
                epoch_val_df.iat[index_number,epoch_val_df.columns.get_loc('target')] = target_label_number
                epoch_val_df.iat[index_number,epoch_val_df.columns.get_loc('predict_label_number')] = predict_label_number
                epoch_val_df.iat[index_number,epoch_val_df.columns.get_loc('predict_logit')] = predict_logit
                epoch_val_df.iat[index_number,epoch_val_df.columns.get_loc('predict_correct')] = target_label_number==predict_label_number
            epoch_val_df = epoch_val_df.dropna()
            epoch_val_df.to_csv(os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_epoch_'+str(epoch)+"_valdf_rank_"+str(rank)+".csv"),index=False)
            
                
                
            
            if rank==0:
                if epoch ==0:
                    best_val_loss = val_loss
                    best_f1 = weighted_f1
                    
                    torch.save(model.module.state_dict(), os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_best_f1+val_loss_model.pth'))
                    torch.save(model.module.state_dict(), os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_best_f1_model.pth'))
                    mlflow_client.log_artifact(mlflow_run_id,os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_best_f1+val_loss_model.pth'),'best_model')        
                    mlflow_client.log_artifact(mlflow_run_id,os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_best_f1_model.pth'),'best_model')        
                    
                    f_best_f1["fold_"+str(k_n)+"_epoch"] = epoch
                    f_best_f1["fold_"+str(k_n)+"_f1score"] = float(best_f1)
                    
                    f_best_f1_valloss["fold_"+str(k_n)+"_epoch"] = epoch
                    f_best_f1_valloss["fold_"+str(k_n)+"_valloss"] = float(best_val_loss)
                    f_best_f1_valloss["fold_"+str(k_n)+"_f1score"] = float(best_f1)
                    
                    # mlflow_client.log_dict(mlflow_run_id,{"fold_"+str(k_n)+"_epoch":epoch,"fold_"+str(k_n)+"_score":float(best_f1)},'fold_'+str(k_n)+'_best_f1.yaml')
                    # mlflow_client.log_dict(mlflow_run_id,{"fold_"+str(k_n)+"_epoch":epoch,"fold_"+str(k_n)+"_loss":float(best_val_loss)},'fold_'+str(k_n)+'_best_f1+valoss.yaml')
                
                ## 베스트 실험결과 저장
                if weighted_f1 > best_f1:
                    best_f1 = weighted_f1
                    torch.save(model.module.state_dict(), os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_best_f1_model.pth'))
                    mlflow_client.log_artifact(mlflow_run_id,os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_best_f1_model.pth'),'best_model')
                    # mlflow_client.log_dict(mlflow_run_id,{"fold_"+str(k_n)+"_epoch":epoch,"fold_"+str(k_n)+"_score":float(best_f1)},'fold_'+str(k_n)+'_best_f1.yaml')
                    f_best_f1["fold_"+str(k_n)+"_epoch"] = epoch
                    f_best_f1["fold_"+str(k_n)+"_f1score"] = float(best_f1)
                    
                    if  val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.module.state_dict(), os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_best_f1+val_loss_model.pth'))
                        mlflow_client.log_artifact(mlflow_run_id,os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_best_f1+val_loss_model.pth'),'best_model')
                        # mlflow_client.log_dict(mlflow_run_id,{"fold_"+str(k_n)+"_epoch":epoch,"fold_"+str(k_n)+"_loss":float(best_val_loss)},'fold_'+str(k_n)+'_best_f1+valoss.yaml')
                        f_best_f1_valloss["fold_"+str(k_n)+"_epoch"] = epoch
                        f_best_f1_valloss["fold_"+str(k_n)+"_valloss"] = float(best_val_loss)
                        f_best_f1_valloss["fold_"+str(k_n)+"_f1score"] = float(best_f1)
                
                

                        
                # 폴드 별 정보 저장
                mlflow_client.log_metric(mlflow_run_id,str(k_n)+"_fold_train_acc",train_acc,step=epoch)
                mlflow_client.log_metric(mlflow_run_id,str(k_n)+"_fold_train_loss",train_loss,step=epoch)
                mlflow_client.log_metric(mlflow_run_id,str(k_n)+"_fold_val_acc",val_acc,step=epoch)
                mlflow_client.log_metric(mlflow_run_id,str(k_n)+"_fold_val_loss",val_loss,step=epoch)
                mlflow_client.log_metric(mlflow_run_id,str(k_n)+"_fold_f1_score",weighted_f1,step=epoch)
                
                train_acc_list[epoch].append(train_acc.view(-1))
                train_loss_list[epoch].append(train_loss)
                val_acc_list[epoch].append(val_acc.view(-1))
                val_loss_list[epoch].append(val_loss)
                f1_socre_list[epoch].append(weighted_f1)
                
                
                
                

    
    if rank==0:
        ## 폴드별에서 최고의 에폭, 로스, f1스코어 정보 저장
        mlflow_client.log_dict(mlflow_run_id,f_best_f1,'each_fold_best_f1.yaml')
        mlflow_client.log_dict(mlflow_run_id,f_best_f1_valloss,'each_fold_best_f1+valoss.yaml')
        
        
        ## 폴드 평균 정보 저장
        train_acc_list = np.mean(train_acc_list,axis=1)
        train_loss_list = np.mean(train_loss_list,axis=1)
        val_acc_list = np.mean(val_acc_list,axis=1)
        val_loss_list = np.mean(val_loss_list,axis=1)
        f1_socre_list = np.mean(f1_socre_list,axis=1)
        
        ## 폴드별 val loss, f1scor 제일 좋은거 선별 
        mlflow_client.log_dict(mlflow_run_id,{"num_fold":int(k_n),"best_f1_score":float(np.max(f1_socre_list)),"epoch":int(np.argmax(f1_socre_list))},'best.yaml')
        for list_index in range(args.env_.epochs):
            mlflow_client.log_metric(mlflow_run_id,"total_train_acc",train_acc_list[list_index],step=list_index)
            mlflow_client.log_metric(mlflow_run_id,"total_train_loss",train_loss_list[list_index],step=list_index)
            mlflow_client.log_metric(mlflow_run_id,"total_val_acc",val_acc_list[list_index],step=list_index)
            mlflow_client.log_metric(mlflow_run_id,"total_val_loss",val_loss_list[list_index],step=list_index)
            mlflow_client.log_metric(mlflow_run_id,"total_f1_score",f1_socre_list[list_index],step=list_index)
        
        
        
        ## hydra 높은값 비교용으로 저장
        save_dict = {"epoch":int(np.argmax(f1_socre_list)),"best_f1_score":float(np.max(f1_socre_list))}
        with open("./temp.pickle",'wb') as fw:
            pickle.dump(save_dict,fw)


    ## val 데이터 합성 -> rank마다 따로 계산된것들 병합하는 작업
    ## mlflow 아티팩트로 저장
    if rank==0:
        
        ## 폴드별, 에폭별, 분산 gpu val 계산 종합
        for k_n in range(0,args.env_.k_fold_n):
            for epoch in range(0, args.env_.epochs):
                total_val_df = pd.read_csv(os.path.join(args.env_.data_save_dir,str(k_n)+'_flod_val_df.csv'),index_col=0)
                for gpu_number in range(0,gpus):
                    ## 각 rank별 계산된것들 val df를 concat함
                    temp_valdf = pd.read_csv(os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_epoch_'+str(epoch)+"_valdf_rank_"+str(gpu_number)+".csv"))
                    if gpu_number==0:
                        val_total_anlydf = temp_valdf
                    else:
                        val_total_anlydf = pd.concat([val_total_anlydf,temp_valdf])
                        
                ## df 합친걸 join하고 정보 저장  
                total_val_df = pd.merge(left = total_val_df , right = val_total_anlydf[['id','predict_label_number','target','predict_correct','predict_logit']], left_on = "id",right_on = "id", how = "left")
                total_val_df.to_csv(os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_epoch_'+str(epoch)+"_val_analysis.csv"),index=False)
                
                ## val 데이터셋에서 맞은수 계산해서 웨이트값 계산
                result_df = pd.DataFrame(list(label_info.keys()),columns=['index'])
                total_val_df = total_val_df[['id','cat3','predict_label_number','target','predict_correct']]
                temp_df1 = total_val_df['cat3'].value_counts().to_frame(name='val_전체수').reset_index()
                temp_df2 = total_val_df.groupby(['cat3'])['predict_correct'].sum().to_frame(name='val_정답수').reset_index()
                result_df = pd.merge(left = result_df , right = temp_df1, left_on = "index",right_on = "index", how = "left")
                result_df = pd.merge(left = result_df , right = temp_df2, left_on = "index",right_on = "cat3", how = "left").drop('cat3',1)
                result_df['val_틀린수'] = result_df['val_전체수'] - result_df['val_정답수']
                result_df['acc'] = result_df['val_정답수'] / result_df['val_전체수']
                result_df = result_df.fillna(0)
                result_df.to_csv(os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_epoch_'+str(epoch)+"_val_label_analysis.csv"),index=False)
                
                ## 폴드별, 에폭별 모델의 검증 acc를 웨이트로 추출한 pikle 파일 저장
                weight_dict = { index:acc for index,acc in zip(result_df['index'],result_df['acc'])}
                with open(os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_epoch_'+str(epoch)+"_val_label_weight.pickle"),'wb') as fw:
                    pickle.dump(weight_dict,fw)

                ## mlflow 아티팩트로 저장
                mlflow_client.log_artifact(mlflow_run_id,os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_epoch_'+str(epoch)+"_val_analysis.csv"),'val_analysis')
                mlflow_client.log_artifact(mlflow_run_id,os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_epoch_'+str(epoch)+"_val_label_analysis.csv"),'val_analysis')
                mlflow_client.log_artifact(mlflow_run_id,os.path.join(args.env_.data_save_dir,'fold_'+str(k_n)+'_epoch_'+str(epoch)+"_val_label_weight.pickle"),'val_analysis')
        
                    
                    
                    
    cleanup()
    
    

@hydra.main(version_base=None, config_path='conf',config_name="config")
def main(config: DictConfig):
    if torch.cuda.device_count() > 1:
      print("병렬 GPU 사용 가능", torch.cuda.device_count())
    
    
    ## hydra config 수정
    hydra_job_num = "(none)"
    if "num" in HydraConfig.get().job:
        hydra_job_num = HydraConfig().get().job.num
    hydra_job_num = str(hydra_job_num) + "_ex"
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.env_.job_name = hydra_job_num
        
        
    ## 결과 임시 저장 폴더
    pathlib.Path(config.env_.data_save_dir).mkdir(parents=True, exist_ok=True)

    
    mp.spawn(trainer, args=(config.env_.gpus, config), nprocs=config.env_.gpus, join=True)
    

    with open('./temp.pickle','rb') as fr:
        best_f1_score = pickle.load(fr)
        
    
    if os.path.exists(config.env_.data_save_dir):
        shutil.rmtree(config.env_.data_save_dir)
    
    return best_f1_score['best_f1_score']
    
if __name__ == '__main__':
    main()