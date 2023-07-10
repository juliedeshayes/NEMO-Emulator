import numpy as np
import matplotlib.pyplot as plt
import math
import random
import joblib
from joblib import Parallel, delayed, parallel_backend
import random


###
import sklearnGPmodel as skgp
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel #, Matérn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, Kernel

#from statsmodels.tsa.statespace.tools import diff


####################################
####################################
##                                ##
##                                ##
##  ANALYSE MODEL FOR NOTEBOOKS:  ##
##         GPYTORCH_MODEL         ##
##         SKLEARNGP_MODEL        ##
##         SKLEARN-OPTI           ##
##                                ##
##                                ##
####################################
####################################

###################
#                 #
#  ANALYSE MODEL  #
#                 #
###################

### 1.1 Get number of pred in the interval

def goodPred(metrics,out=5,f=1.96,evaluation_method="std",w=16):
    if evaluation_method == "dist":
        l = [1 if dist<maxi*f else 0 for dist,maxi in zip(abs(metrics["dist"]),abs(metrics["max"]))]   
        
    if evaluation_method == "std":
        l = [1 if p < m + f*np.sqrt(std) and p > m - f*np.sqrt(std) else 0 for p,m,std in
             zip(metrics["ma_pred"],metrics["ma_true"],metrics["std_true"])]

    else:
        print("Method not found")
        return -1
    
    zero = [ind for ind, val in enumerate(l) if val == 0]
    for i in range(len(zero)-out): 
        if (zero[i]+out - zero[i+out]) == 0:
            return len(l[:zero[i]])+w//2

    return len(l)+w//2


def goodPredSTD(eof,train,gp):
    random.seed(10)
    mean_preds,std_preds,metrics = skgp.predict(eof,train,gp,1,show=False)
    correct_years = goodPred(metrics,out=5,f=1.96,evaluation_method="std")
    return correct_years


### Analyse results


def countResult(result):
    return np.array([np.count_nonzero(result==i) for i in np.unique(result)])

def ClassResult(res,bornes):
    r = np.empty_like(res)
    r[res == 0 ] = 0
    r[(res > 0) & (res <= bornes[0])] = 1
    for i in range(len(bornes)-1):
        r[(res > bornes[i]) & (res <= bornes[i+1])] = i+2
    r[(res > bornes[i+1])] = i+3
    return r,countResult(r)


def ClassResult2(res,bornes):
    r = np.empty_like(res)
    r[res == 0 ] = 0
    r[(res > 0) & (res <= bornes[0])] = 1
    r[res > bornes[0]] = 2
    return r,countResult(r)



#######################
#                     #
#   OPTIMISE KERNEL   #
#                     #
#######################


#Create a list of possible kernels 

def createKernelCombination(r=RBF(),n=4):
    k=[RBF(), RationalQuadratic()] 
    if(n==1):
        return r
    else:
        return createKernelCombination(r+k[0],n=n-1),createKernelCombination(r*k[0],n=n-1),createKernelCombination(r+k[1],n=n-1),createKernelCombination(r*k[1],n=n-1)

    
def listOfKernels(r=RBF(),n=4):
    return np.array(createKernelCombination(r,n)).reshape(-1)


def getAllGP(n=4,r=RBF()):    
    kernels=createKernelCombination(r,n)  
    listGP=[]
    for kernel in np.array(kernels).reshape(-1):
        kernel=kernel+WhiteKernel()
        listGP.append(GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=0))
    return listGP

# Select the best kernel regarding diff test, and simu

def BestKernelsSTD(sig,model,tests):
    score = Parallel(n_jobs=10)(delayed(goodPredSTD)(sig,i,model) for i in tests)
    return np.sum(np.array(score))


def findBestKernelsSTD(simulations,models,list_test,min_test=50,n_jobs=1):
    score  = []
    result = []
    for simu in simulations:
        max_len  = np.shape(simu)[0] - min_test
        maxi     = np.searchsorted(list_test, max_len)
        test_cut = list_test[:maxi]
        result.append(Parallel(n_jobs)(delayed(analyse.BestKernelsSTD)(simu,m,test_cut) for m in models))
    result = np.sum(result,axis=0)
    return sorted([(m,r) for m,r in zip(models,result)], key=lambda item:item[1],reverse=True)  

