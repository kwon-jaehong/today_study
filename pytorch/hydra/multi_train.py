from http import client
import random
import torch
import os
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import mlflow
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf,open_dict

from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID,MLFLOW_RUN_NAME

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

def train(a):


    return 1 
        
def val(n):        
    pass

    return 2

def trainer(rank, gpus, args):

    setup(rank, gpus)
    seed_everything(41) # Seed 고정
    
    if rank ==0:
        mlflow_client = MlflowClient()
        mlflow_run = mlflow_client.create_run(experiment_id='0')
        
        mlflow_run.info
        mlflow_run_id = mlflow_run.info.run_id

    
    for epoch in range(0, 5):
        lr = train(1)
        val_acc= val(2)
        
        if rank == 0:
            client.log_params(mlflow_run_id,{"temp":0.1})       
            client.log_params(mlflow_run_id,{"temp":0.1})    
            # mlflow.log_metric("lr",lr,epoch)      


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
        config.job_name = hydra_job_num
        
    
 
    
    mp.spawn(trainer, args=(2, config), nprocs=2, join=True)
   
    
    temp = 1

    
    return 1
    
if __name__ == '__main__':
    main()