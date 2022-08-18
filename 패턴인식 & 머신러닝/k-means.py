import numpy as np
import random
import matplotlib.pyplot as plt
 
np.random.seed(100)
num_data = 50
 
x11 = np.linspace(0.3,0.7,20)
x12 = np.linspace(1.3,1.8,15)
x13 = np.linspace(2.4,3,15)
x1 = np.concatenate((x11,x12,x13),axis=None)
error = np.random.normal(1,0.5,num_data)
x2 = 1.5*x1+2+error
fig = plt.figure(figsize=(7,7))
fig.set_facecolor('white')
plt.scatter(x1, x2, color='k')
plt.show()


def kmeans_clustering(X, n_clusters, init_center=None, max_iter=10, epsilon=1e-4, random_state=100): 
    # 센터값 초기화
    init_center = None
    if init_center is None:
        random.seed(random_state)
        idx = random.sample(range(X.shape[0]), n_clusters)
        center = X[idx,:]
    else:
        center = init_center
    iteration = 1
    
    labels_history = [] # label history 
    center_history = [] # centeroid history
    while(iteration<=max_iter):
        ## assign label
        labels = []
        for i in range(X.shape[0]):
            ## 행렬의 한개만 데이터 불러옴
            data = X[i, :]
            
            ## 데어터를 클러스터 센터값이랑 비교했을때 가장 가까운곳에 레이블을 부여함
            labels.append(np.argmin([np.linalg.norm(data-x) for x in center]))
        
        # list자료형을 np
        labels = np.array(labels)
        
        
        ## 클러스터 센터값 업데이트
        next_center = []
        for i in range(n_clusters):
            
            # 0번째 기준, 타겟 idx를 저장()
            target_idx = np.where(labels==i)[0]
            
            # 그 자료들의 평균값을 구함 -> 업데이트할 좌표값임
            center_val = np.mean(X[target_idx,:], axis=0)
            next_center.append(center_val)
 
 
        next_center = np.array(next_center)
        ## 엡실론 == 학습 목표치 이상 달성했다면 == 업데이트된 좌표가, 이전 클러스터 좌표와 차이가 없다면
        if epsilon:
            if np.linalg.norm(next_center-center) <= epsilon:
                break
            
        center = next_center
        labels_history.append(labels)
        center_history.append(center)
        iteration += 1
    return (labels, iteration, labels_history, center_history)


X = np.stack([x1, x2], axis=1)
 
init_center= np.array([[2,4],[1,5],[2.5,6]])
max_iter=50
epsilon=1e-10
random_state=101
n_clusters=3
results = kmeans_clustering(X, n_clusters, init_center, max_iter, epsilon=1e-4, 
                           random_state=100)
labels = results[0]