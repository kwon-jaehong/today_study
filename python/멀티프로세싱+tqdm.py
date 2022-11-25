
# from multiprocessing import Pool

from tqdm import *

from torch import multiprocessing as mp

def f(x):
    return x

## Pool에 아무것도 선언 하지않하면 CPU 최대로 설정이됨
with mp.Pool() as p:
    max = 10000
    with tqdm(total=max) as pbar:
        for _ in p.imap_unordered(f,range(0,max)):
            pbar.update()



print(1)