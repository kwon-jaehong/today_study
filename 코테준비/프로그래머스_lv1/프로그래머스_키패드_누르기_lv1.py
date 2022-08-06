# def distance
from numpy import argmin

def distance_compare(position_r,position_l,target_position,hand):
    r_y,r_x = position_r
    l_y,l_x = position_l
    
    t_y,t_x = target_position
    
    ## 왼손, 오른손으로 눌럿을때 거리값
    dis_r = abs(r_x-t_x)+abs(r_y-t_y)
    dis_l = abs(l_x-t_x)+abs(l_y-t_y)
    
    
    hand_result = "None"
    if dis_l == dis_r and t_x==1:
        if hand == "right":
            position_r = target_position
            hand_result = "R"
        else:
            position_l = target_position
            hand_result = "L"
    elif t_x==0:
        position_l = target_position
        hand_result = "L"
        
    elif t_x==2:
        position_r = target_position
        hand_result = "R"
        
    else:
        flg = argmin([dis_l,dis_r])
        if flg==0:
            position_l = target_position
            hand_result = "L"
        else:
            position_r = target_position 
            hand_result = "R"
            
            
    return [position_r,position_l,hand_result]
def solution(numbers, hand):

    hand_map = {}
    r_init = "#"
    l_init = "*"    
    var = 1
    for i in range(0,3):
        for j in range(0,3):
            hand_map[var]=[i,j]
            var +=1
    hand_map["*"] = [3,0]
    hand_map[0] = [3,1]
    hand_map["#"] = [3,2]
    
    
    position_r = hand_map[r_init]
    position_l = hand_map[l_init]
    
    

    
    result_str = ""
    
    for key in numbers:
        position_r,position_l,temp_str = distance_compare(position_r,position_l,hand_map[key],hand)
        result_str += temp_str

    answer = result_str
    return answer



solution([1, 3, 4, 5, 8, 2, 1, 4, 5, 9, 5], "right")
# [1, 3, 4, 5, 8, 2, 1, 4, 5, 9, 5], "right"
# 기댓값 〉	"LRLLLRLLRRL"