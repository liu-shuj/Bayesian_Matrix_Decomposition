from typing import Union,Tuple
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse

def VB_Decomp(M:Union[csr_matrix,np.ndarray],rank:int,maxiter:int=100)->Tuple[np.ndarray,np.ndarray]:
    # init
    I=M.shape[0]
    J=M.shape[1]
    n=rank
    sigma_sq = np.ones(n)
    rho_sq = np.ones(n) / n
    tau_sq=1
    u_bar=[]
    v_bar=[]
    t=[]
    S,Phi,Psi=[],[],[]

    for i in range(0,I):
        Phi.append(np.eye(n))
        u_bar.append(np.random.normal(0,1,n))
    for j in range(0,J):
        Psi.append(np.eye(n))
        S.append(np.diag(1/rho_sq))
        t.append(np.zeros(n))
        v_bar.append(np.random.normal(0,1,n))

    Phi=np.array(Phi)
    Psi=np.array(Psi)
    S=np.array(S)
    t=np.array(t)
    u_bar=np.array(u_bar)
    v_bar=np.array(v_bar)

    norm_u=0
    norm_v=0

    N=[]
    for i in range(0, I):
        N.append(scipy.sparse.find(M[i])[1])
    ob = scipy.sparse.find(M)

    #--------------------------------------------------------------------------------------------
    # EM iteration
    for iter in range(0,maxiter):
    # E step
        # update Q(u_i)
        for i in range(0,I):
            outer=np.zeros((n,n))
            for j in N[i]:
                outer+=np.outer(v_bar[j],v_bar[j])
            Phi[i]=np.linalg.inv(np.diag(1/sigma_sq)+(Psi[N[i]].sum(0)+outer)/tau_sq)
            u_bar[i]=Phi[i].dot(((M[i,N[i]]*(v_bar[N[i]]))/tau_sq).sum(0))
            S[N[i]]+=(Phi[i]+np.outer(u_bar[i],u_bar[i]))/tau_sq
            t[N[i]]+=(np.outer(csr_matrix(M[i,N[i]]).toarray(),(u_bar[i]))/tau_sq)

        #update Q(v_j)
        Psi=np.linalg.inv(S)
        for j in range(0,J):
            v_bar[j]=Psi[j].dot(t[j])

    # M step
        for l in range(0,n):
            sigma_sq[l]=((Phi[:,l,l]+u_bar[:,l]**2).sum())/(I-1)

        K=len(ob[1])
        Tr=0
        for i,j in np.array([ob[0],ob[1]]).T:
            A = Phi[i] + np.outer(u_bar[i], u_bar[i])
            B = Psi[j] + np.outer(v_bar[j], v_bar[j])
            Tr+=np.trace(A.dot(B))
        tau_sq=(((ob[2]**2)-(2*ob[2]*np.einsum('ij,ij->i',u_bar[ob[0]],v_bar[ob[1]]))).sum()+Tr)/(K-1)

	# calc norm
        cur_norm_u=np.linalg.norm(u_bar)
        cur_norm_v=np.linalg.norm(v_bar)
        if(abs(cur_norm_u-norm_u)<0.01 or abs(cur_norm_v-norm_v)<0.01):
            break
        else:
            norm_u,norm_v=cur_norm_u,cur_norm_v
    return np.array(u_bar),np.array(v_bar)


# Generator version, in convenience for evaluating performance per iteration
# exactly the same except 'return' is replaced by 'yield'
def VB_Decomp_Gen(M:Union[csr_matrix,np.ndarray],rank:int,maxiter:int=100)->Tuple[np.ndarray,np.ndarray]:
    # init
    I=M.shape[0]
    J=M.shape[1]
    n=rank
    sigma_sq = np.ones(n)
    rho_sq = np.ones(n) / n
    tau_sq=1
    u_bar=[]
    v_bar=[]
    t=[]
    S,Phi,Psi=[],[],[]

    for i in range(0,I):
        Phi.append(np.eye(n))
        u_bar.append(np.random.normal(0,1,n))
    for j in range(0,J):
        Psi.append(np.eye(n))
        S.append(np.diag(1/rho_sq))
        t.append(np.zeros(n))
        v_bar.append(np.random.normal(0,1,n))

    Phi=np.array(Phi)
    Psi=np.array(Psi)
    S=np.array(S)
    t=np.array(t)
    u_bar=np.array(u_bar)
    v_bar=np.array(v_bar)

    norm_u=0
    norm_v=0

    N=[]
    for i in range(0, I):
        N.append(scipy.sparse.find(M[i])[1])
    ob = scipy.sparse.find(M)

    # EM iteration
    for iter in range(0,maxiter):
    # E step
        # update Q(u_i)
        for i in range(0,I):
            outer=np.zeros((n,n))
            N_i=N[i]
            for j in N_i:
                outer+=np.outer(v_bar[j],v_bar[j])
            Phi[i]=np.linalg.inv(np.diag(1/sigma_sq)+(Psi[N_i].sum(0)+outer)/tau_sq)
            mtplr=((M[i,N_i]*(v_bar[N_i]))/tau_sq).sum(0)
            u_bar[i]=Phi[i].dot(mtplr)
            S[N_i]+=(Phi[i]+np.outer(u_bar[i],u_bar[i]))/tau_sq
            t[N_i]+=(np.outer(csr_matrix.toarray(M[i,N_i]),(u_bar[i]))/tau_sq)

        #update Q(v_j)
        Psi=np.linalg.inv(S)
        for j in range(0,J):
            v_bar[j]=Psi[j].dot(t[j])

    # M step
        for l in range(0,n):
            sigma_sq[l]=((Phi[:,l,l]+u_bar[:,l]**2).sum())/(I-1)

        K=len(ob[1])
        Tr=0
        for i,j in np.array([ob[0],ob[1]]).T:
            A = Phi[i] + np.outer(u_bar[i], u_bar[i])
            B = Psi[j] + np.outer(v_bar[j], v_bar[j])
            Tr+=np.trace(A.dot(B))
        tau_sq=(((ob[2] ** 2) - (2 * ob[2] * np.einsum('ij,ij->i',u_bar[ob[0]],v_bar[ob[1]]))).sum()+Tr)/(K-1)

        cur_norm_u=np.linalg.norm(u_bar)
        cur_norm_v=np.linalg.norm(v_bar)
        if(abs(cur_norm_u-norm_u)<0.01 or abs(cur_norm_v-norm_v)<0.01):
            break
        else:
            norm_u,norm_v=cur_norm_u,cur_norm_v
        yield np.array(u_bar),np.array(v_bar)