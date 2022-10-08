# https://github.com/optuna/optuna-examples/blob/main/hydra/simple.py
# https://github.com/ashleve/lightning-hydra-template/blob/main/configs/hparams_search/mnist_optuna.yaml
# https://www.youtube.com/watch?v=W_mJxQupJ4I&t=94s
## 옵투나 베이지안 으로
 
## 병렬 실행이 아님, -> 한 태스크가 끝나고 다른 테스크가 시작되는 거임

from omegaconf import DictConfig
import hydra
import mlflow
import time

@hydra.main(version_base=None, config_path='conf',config_name="config")
def main(config: DictConfig) -> None:
    with mlflow.start_run():
        time.sleep(5)
        mlflow.log_params(config)
    return 0        

    
if __name__== "__main__":
	main()