
# from multiprocessing import Pool

from tqdm import *
from functools import partial
from itertools import repeat

from torch import multiprocessing as mp



def f(x,y):
    global list_a
    print("프로세스",mp.current_process()," == ",list_a)
    return x

## Pool에 아무것도 선언 하지않하면 CPU 최대로 설정이됨
def main():
    global list_a
    list_a = [1,2,3,4,5]
    with mp.Pool() as p:
        max = 10000
        with tqdm(total=max) as pbar:
            for _ in p.starmap(f,zip(range(0,max),range(0,max))):
                pbar.update()

if __name__ == "__main__":
    main()