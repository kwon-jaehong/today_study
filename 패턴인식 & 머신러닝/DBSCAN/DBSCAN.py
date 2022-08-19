# ① DBSCAN은 임의로 고른(가 본 적 없는) starting data point에서 시작한다. 이 point의 이웃은 거리 입실론 ε을 사용해 추출된다.

# ② 만약 '이웃' 안에 충분한 수의 points(minPoints에 따라)가 들어있다면 클러스터링을 시작한다. 새로운 클러스터에 있는(현재 시작점인) 첫 번째 point와 다르게 나머지 points는 noise로 라벨링된다. 나중에 noise도 클러스터의 부분이 된다. 그리고 두 경우 다 방문했다는 것을 기억한다.

# ③ 새로운 클러스터의 first point를 위해 입실론 거리에 있는 이웃에 속한 points도 같은 클러스터의 일부가 된다. 이러한 절차는 클러스터 그룹에 추가된 적이 있는 모든 새로운 points를 위해 반복된다.

# ④ 2, 3번은 모든 points가 정의될 때까지 계속된다. 즉, 모든 입실론 이웃인 points이 라벨링 될 때 까지 계속된다.

# ⑤ 현재 클러스터링이 끝나면 새로운 방문하지 않은 point가 나오고 다시 위 과정이 반복된다. 이는 더 먼 거리에 있는 cluster와 noise를 발견하기 위함이다. 이 과정도 모든 points를 방문할 때까지 계속된다.

import numpy as np

## 2차원 이상일 경우 np로 계산
def distance (a,b):
    return abs(a-b)

## 탐색할 거리
Epsilon = 5

## 최소 군집 갯수 2 -> 군집영역이 2보다 작으면 노이즈로 봄

x = [25, 25, 25, 25, 25, 25, 25, 25, 25, 317, 317, 317, 317, 317, 317, 317, 317, 317, 610, 610, 610, 610, 610, 610, 610, 610, 610, 902, 902, 902, 902, 902, 902, 902, 902, 902, 1192, 1192, 1192, 1192, 1192, 1192, 1192, 1192, 1192, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1775, 1775, 1775, 1775, 1775, 1775, 1775, 1775, 1775, 2066, 2066, 2066, 2066, 2066, 2066, 2066, 2066, 2066, 2358, 2358, 2358, 2358, 2358, 2358, 2358, 2358, 2358, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2943, 2943, 2943, 2943, 2943, 2943, 2943, 2943, 2943, 3237, 3237, 3237, 3237, 3237, 3237, 3237, 3237, 3237, 3527, 3527, 3527, 3527, 3527, 3527, 3527, 3527, 3527, 3821, 3821, 3821, 3821, 3821, 3821, 3821, 3821, 3821, 0]

## [값,라벨,검사유무,index] 초기화
x = np.array([ [var,-1,False,i ] for i,var in enumerate(x)])

# init list , 큐 역활
calculate_list = []

## 첫번째 값부터 라벨 시작, 검색 여부
label_pointer = 0
x[0][2] = True
## 라벨 포인터
x[0][1] = label_pointer
calculate_list.append(x[0])

## 조건문 
while len(calculate_list) > 0:
    serch_var = calculate_list.pop(0)
    
    ## 거리계산값, 검색 여부의 index만 뽑아옴
    candidate_index_list = [ i for var,label,flg,i in x if flg==False and distance(serch_var[0],var) < Epsilon ]
    # candidate_index_list = []
    # for i,item in enumerate(x):
    #     if item[2]==False:
    #         if distance(serch_var[0],item[0]) < e_var:
    #             candidate_index_list.append(i)    
    
    
    for index in candidate_index_list:
        ## 정보 바꾸어주고
        x[index][2] = True
        x[index][1] = label_pointer
        
        ## 검색할 list에 더함
        calculate_list.append(x[index])
    
    
    ## 다 검사하지 않았다면 검색 안된 x배열의 정보를 계산(큐) 배열에 추가
    if len(calculate_list)==0 and np.all(x[:,2]==1)==False:
        label_pointer +=1
        _,index_arr = np.where([x[:,2]==0])
        
        x[index_arr[0]][2] = True
        x[index_arr[0]][1] = label_pointer
                
        calculate_list.append(x[index_arr[0]])
        
x = x[:,0:2]
## 이후에는 라벨링 카운팅을 이용하여 라벨이 안된정보는 삭제하거나 노이즈로 라벨링을 돌림







