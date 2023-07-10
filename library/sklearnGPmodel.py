import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel #, Matérn
from sklearn.gaussian_process import GaussianProcessRegressor
import random
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_squared_error,r2_score


#########################################
#########################################
##                                     ##
##                                     ##
##  SKLEARNGPMODEL LIB FOR NOTEBOOKS:  ##
##              SKLEARN_GP             ##
##                                     ##
##                                     ##
#########################################
#########################################


######################
#                    #
#  FORECAST ONE EOF  #
#                    #
######################

### SELECT PARAMETERS ###

def defineGP():
    long_term_trend_kernel = 50.0**2*RBF(length_scale=100.0)
    irregularities_kernel  = 0.5**2*RationalQuadratic(length_scale=5.0, alpha=1.0)
    noise_kernel           = 0.1**2*RBF(length_scale=0.01) + 2*WhiteKernel(noise_level=1)
    kernel = long_term_trend_kernel + irregularities_kernel + noise_kernel
    return GaussianProcessRegressor(kernel=kernel, normalize_y=False,n_restarts_optimizer=1)

### APPLY MODEL ###

def standardize(signal,*args):
    if len(args) == 0:
        m   = np.nanmean(signal, axis=None, keepdims=True)
        std = np.nanstd(signal,  axis=None, keepdims=True)
    else:
        m, std = args[0], args[1]
    return (signal - m) / (2.0 * std), m, std


def getMetrics(ytest,ypred,w):
    ma_test = np.convolve(ytest/w,              np.ones(w), mode="valid")
    ma_pred = np.convolve(ypred/w,              np.ones(w), mode="valid")
    dist    = np.convolve((ytest-ypred)/w,      np.ones(w), mode="valid")
    mse     = np.convolve(((ytest-ypred)**2)/w, np.ones(w), mode="valid")
    
    dist_max,std=[],[]
    for i in range(w,len(ypred)+1):
        windowT = ytest[i-w:i]
        windowP = ypred[i-w:i]
        #maxi/mini
        maxi = np.max(windowT)-np.mean(windowT)
        mini = np.mean(windowT)-np.min(windowT)
        dist_max.append(max(maxi,mini))
        #std
        std.append(np.std(windowT,ddof=1))

    return {"ma_true":ma_test,"ma_pred":ma_pred,"dist":dist, "dist_max":dist_max,"mse":mse,"std_true":np.array(std)}


### REGRESSION PREDICTION ###
def showPred(eof,train_len,steps,pred_mean,pred_std):
    xtrain = np.arange(train_len)
    xtest  = np.arange(train_len,len(eof))
    prediction_stop = len(eof)+steps
    xpred  = np.arange(train_len,prediction_stop)
    
    figure = plt.figure(figsize=(12,5))
    plt.plot(xtrain, eof[:train_len], linestyle="dashed", color="black", alpha=0.8,label = "Train serie")
    
    plt.plot(xpred,  pred_mean  , color="orange", label = "GP forecast")
    plt.fill_between(
        xpred,
        pred_mean[:] - pred_std,
        pred_mean[:] + pred_std,
        color = "orange",
        alpha = 0.2)  
    
    plt.plot(xtest,  eof[train_len:], color="black", linestyle="dashed", alpha =0.5,label = "Test serie")

    plt.title("Forecasting first eof")
    plt.legend()
    plt.show()
    

def predict(eof,train_len,process,steps,norm=True,show=True,w=16):
    ytrain = eof[:train_len]
    xtrain = np.arange(train_len).reshape(-1, 1) #format [[0],[1],...]    
    
    if norm:
        ytrain,m,std = standardize(ytrain) 
        
    process.fit(xtrain, ytrain)   
    
    prediction_stop  = len(eof)+steps
    xtest            = np.arange(train_len,prediction_stop).reshape(-1, 1) 
    pred_mean,pred_std = process.predict(xtest,return_std=True) 
    
    if norm:
        pred_mean  = pred_mean * 2 * std + m
        pred_std   = pred_std  * 2 * std
    
    if show: 
        showPred(eof,train_len,steps,pred_mean,pred_std)

    metrics = getMetrics(eof[train_len:],pred_mean[:-steps],w)
    return pred_mean,pred_std,metrics


### AUTO-REGRESSION PREDICTION ###


def showpredAuto(xtest,ytest,ypred,stdpred):
    figure = plt.figure(figsize=(12,5))
    axis=np.arange(len(xtest),len(xtest)+len(ytest))
    plt.plot(xtest, linestyle="dashed", label = "Train serie")
    plt.plot(axis,ytest,linestyle="dashed",color="grey",label="Test serie")
    plt.plot(axis,ypred,color ="tab:blue",label="Forcasted serie")
    plt.fill_between(
        axis,
        ypred - stdpred,
        ypred + stdpred,
        color = "tab:blue",
        alpha = 0.2)
    plt.legend()
    plt.show()

        
def predictAuto(eof,process,train_len,nlags,steps,norm=True,show=True,w=16):
    train  = np.array(eof[:train_len-nlags-steps])
    xtrain = np.array([eof[i       : i+nlags]        for i in range(len(train))])
    ytrain = np.array([eof[i+nlags : i+nlags+steps]  for i in range(len(train))])
    
    xtest  = np.array(eof[train_len-nlags:train_len]).reshape(1,-1)
    ytest  = np.array(eof[train_len:train_len+steps]).reshape(1,-1)
    
    if norm:
        xtrain,m,std = skgp.standardize(xtrain)
        ytrain       = ytrain-m/(2*std)
        xtest        = xtest-m/(2*std)
        
    process.fit(xtrain, ytrain)
    pred_mean,pred_std = process.predict(xtest,return_std=True)

    if norm:
        pred_mean  = pred_mean * 2 * std + m
        pred_std   = pred_std  * 2 * std
        print("resultats faux mauvaise normalisation mettre norm = False")
    
    if show:
        showpredAuto(xtest.squeeze(),ytest.squeeze(),pred_mean.squeeze(),pred_std.squeeze())

    metrics = getMetrics(ytest.squeeze(),pred_mean.squeeze(),w)
    
    return pred_mean.squeeze(),pred_std.squeeze(),metrics


"""

def predictAuto(eof,process,train_len,nlags,steps,norm=True,show=3,w=15):
    train        = np.array(eof[:train_len-nlags-steps])
   # indiceMax    = len(train)-nlags-steps
    
    xtrain = np.array([eof[i       : i+nlags]        for i in range(len(train))])
    ytrain = np.array([eof[i+nlags : i+nlags+steps]  for i in range(len(train))])
    
    if norm:
        xtrain,m,std = skgp.standardize(xtrain)
        ytrain       = ytrain-m/(2*std)
    
    process.fit(xtrain, ytrain)
    
    xtest      = np.array(eof[train_len-nlags:train_len]).reshape(1,-1)
    ytest      = np.array(eof[train_len:train_len+steps]).reshape(1,-1)
    #indiceMax = len(test)-nlags-steps
    #xtest     = np.array([test[i        : i+nlags]      for i in range(indiceMax)])
    #ytest     = np.array([test[i+nlags : i+nlags+steps] for i in range(indiceMax)])
    
    if norm:
        xtest       = xtest-m/(2*std)

    pred_mean,pred_std = process.predict(xtest,return_std=True)

    if norm:
        pred_mean  = pred_mean * 2 * std + m
        pred_std   = pred_std  * 2 * std
    
    if norm:
        print("resultats faux mauvaise normalisation mettre norm = False")
    
   # if show > 0:
   #     showpredAuto(show,ytest,pred_mean,pred_std)

    metrics = getMetrics(ytest[0],pred_mean,w)
    
    return ytest,pred_mean,pred_std,metrics


    if get_scores and n_test>0:
        scores = sliding_metrics(np.array(signal_test),mean_y_pred[:n_test],w)
        return mean_y_pred, std_y_pred,scores
    else:
        return mean_y_pred, std_y_pred
    
    
    
def compare_output(simu,sig_spin):
    figure = plt.figure(figsize=(15,7))

    plt.plot(np.arange(len(sig_spin)), sig_spin, color="gray", linestyle="dashed", alpha=1,label="Spin")
    plt.plot(np.arange(len(simu)), simu, color="red", linestyle="dashed",alpha=0.6, label="Measurements")

    plt.legend()
    plt.show()



def confInt(arr,prob,w):
    std     = np.std(arr, ddof=1)
    t_value = t.ppf((1-prob)/2,w-1)
    error   = t_value*(std/np.sqrt(w))
    return error

    
def sliding_metrics(y_true, y_pred, w):
    ma_true = np.convolve(y_true/w, np.ones(w),mode="valid")
    ma_pred = np.convolve(y_pred/w, np.ones(w),mode="valid")
    
    dist = np.convolve((y_true-y_pred)/w, np.ones(w),mode="valid")
    
    mse = np.convolve(((y_true-y_pred)**2)/w, np.ones(w),mode="valid")
    
    acc,dist_max,err,std=[],[],[],[]
    for i in range(w,len(y_pred)+1):
        windowT = y_true[i-w:i]
        windowP = y_pred[i-w:i]
        #maxi/mini
        maxi = np.max(windowT)-np.mean(windowT)
        mini = np.mean(windowT)-np.min(windowT)
        dist_max.append(max(maxi,mini))
        #acc
        wT = windowT - np.mean(windowT)
        wP = windowP - np.mean(windowP)
        acc.append(np.mean(np.sum(wT*wP)/np.sqrt(np.sum(wT**2)*np.sum(wP**2))))
        #conf int
        err.append(confInt(windowT,0.95,w))
        #std
        std.append(np.std(windowT,ddof=1))
    metrics = {"ma_true":ma_true,"max":np.array(dist_max),"ma_pred":ma_pred,
               "dist":dist,"mse":mse,"acc":np.array(acc),
               "confInt95":np.array(err),"std_true":np.array(std)}
    return metrics


def show_GP_error(signal_test, mean_y_pred):
    figure = plt.figure(figsize=(15,3))
    error  = [abs((p-s)//p)*100 for s,p in zip(signal_test,mean_y_pred)]
    plt.plot(error)
    plt.title("Error on train set");
    
    
def regen_new_map_from_regression(X_input, X_pred, mask,pca, m, std):
    last_point = np.concatenate([X_pred[-1:], X_input[-1, 1:]], axis=0)  
    map_       = np.zeros((331, 360), dtype=float)
    bool_mask  = mask.cpu().numpy()
    
    map_[bool_mask]  = pca.inverse_transform(last_point)
    map_[~bool_mask] = np.nan
    
    map_ = 2 * map_ * std + m
    return map_






"""