# https://github.com/optuna/optuna-examples/blob/main/hydra/simple.py
# https://github.com/ashleve/lightning-hydra-template/blob/main/configs/hparams_search/mnist_optuna.yaml
# https://www.youtube.com/watch?v=W_mJxQupJ4I&t=94s
## 옵투나 베이지안 으로
 
## 병렬 실행이 아님, -> 한 태스크가 끝나고 다른 테스크가 시작되는 거임

from omegaconf import DictConfig
import hydra
import mlflow
import time
from hydra.core.hydra_config import HydraConfig

@hydra.main(version_base=None, config_path='conf',config_name="config")
def main(config: DictConfig) -> None:
    
    ## hydra 멀티중 하이드라 job 넘버 받는부분
    hydra_job_num = "알수없음_"
    if "num" in HydraConfig.get().job:
        hydra_job_num = HydraConfig().get().job.num
    hydra_job_num = str(hydra_job_num) + "번째_실험"
    
    
    with mlflow.start_run(run_name=hydra_job_num):
        time.sleep(2)
        mlflow.log_params(config)


        
        mlflow.log_dict({"num_epoch":hydra_job_num},'temp.json')
            
    return 0        

    
if __name__== "__main__":
	main()