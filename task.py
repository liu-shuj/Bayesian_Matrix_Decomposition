from scipy.sparse import lil_matrix
from core.Var_Bayes_Decomp import *
import scipy.sparse
import time

dataset="hw2data/data-500-500-3.txt"
rank_list=[1,2,3,5,10,20]
part1_rank=3

with open(dataset) as f:
    line=f.readline()
    I,J,K=map(int,line.split()) # number of users, items, scores
    M=lil_matrix((I,J))
    testset=[]
    half=int(K/2)
    for i in range(0,half):
        data=f.readline().split()
        x=int(data[0])
        y=int(data[1])
        M[x,y]=float(data[2])
    for i in range(0,K-half):
        data=f.readline().split()
        x=int(data[0])
        y=int(data[1])
        testset.append((x,y,float(data[2])))

M=M.tocsr()

# baseline
a,b=[],[]
c=scipy.sparse.find(M)[2].mean()
for i in range(0,M.shape[0]):
    if(len(scipy.sparse.find(M[i])[0])==0):
        a.append(c)
    else:
        a.append(M[i].mean())
for j in range(0,M.shape[1]):
    if(len(scipy.sparse.find(M.T[j])[0])==0):
        b.append(c)
    else:
        b.append(M.T[j].mean())

real_scores,predicted_baseline=[],[]
for example in testset:
    i,j,score=example
    real_scores.append(score)
    predicted_baseline.append((a[i]+b[j])/2)
RMSE_baseline=np.sqrt(((np.array(real_scores)-np.array(predicted_baseline))**2).mean())
print("RMSE of baseline:")
print(RMSE_baseline)

# part 1
RMSE_iter=[]
time_record=[]
last_time=time.time()
total_time=0
for u, v in VB_Decomp_Gen(M, part1_rank):
    total_time+=time.time()-last_time
    time_record.append(total_time)
    predicted_scores=[]
    for example in testset:
        i, j, score = example
        predicted_scores.append(u[i].dot(v[j]))
    RMSE_iter.append(np.sqrt(((np.array(real_scores)-np.array(predicted_scores))**2).mean()))
    last_time=time.time()

print("RMSE after per iteration:")
print(RMSE_iter)
print("Total time after per iteration:")
print(time_record)

# part 2
RMSE_rank=[]
for rank in rank_list:
    u,v=VB_Decomp(M,rank)
    predicted_scores=[]
    for example in testset:
        i, j, score = example
        predicted_scores.append(u[i].dot(v[j]))
    RMSE_rank.append(np.sqrt(((np.array(real_scores)-np.array(predicted_scores))**2).mean()))
print("RMSE at rank={rank_list},respectively:".format(rank_list=rank_list))
print(RMSE_rank)
