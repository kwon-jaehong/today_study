stages = [2, 1, 2, 6, 2, 4, 3, 3]
N = 5
 
fail_rate =  [ [stage,0] for stage in range(0,N+1)]

clear_player = len(stages)
for i in range(0,N+1):
    clear_player -= stages.count(i-1)
    
    if stages.count(i) != 0:
        fail_rate[i][1] = stages.count(i) / clear_player
        # print(stages.count(i) , clear_player)
    else:
        fail_rate[i][1] = 0

del fail_rate[0]

print([ key for key,var in sorted(fail_rate,key=lambda x:x[1],reverse=True)])
    # print(stage_pointer)


# answer = []

# return answer

# result 길이는 stage max값 -1

# solution(5, [2, 1, 2, 6, 2, 4, 3, 3])
# [3, 4, 2, 1, 5]