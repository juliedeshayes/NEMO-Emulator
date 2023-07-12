import sys 
sys.path.insert(0, "/gpfswork/rech/omr/uen17sn/NewSpinUp/lib")
import prepare

import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
from joblib import Parallel, delayed, parallel_backend

##########################################
##########################################
##                                      ##
##  STRATIFICATION LIB FOR NOTEBOOKS:   ##
##       LOAD_SIMU_GRIDS                ##
##       PREPARE_DATASET                ##
##                                      ##
##                                      ##
##########################################
##########################################


#######################
#  Apply acp on simu  #
#  methods 1, 2 & 4   #
#######################

def get_nan_values(array): 
    bool_mask = np.asarray(np.isfinite(array), dtype=bool)
    return bool_mask 

def apply_nan_mask(var):
    mask = get_nan_values(var[0])
    return var[:,mask],mask

def toTimeSeries(array,n_comp=0.9,pca=None):
    array,mask = apply_nan_mask(array)
    
    if pca is not None:
        array = pca.transform(array)
        return array,mask
    
    else:
        pca = PCA(n_comp, whiten=False)
        array = pca.fit_transform(array)
        return array,pca,mask
    return 0
    
def plot_all(dico,array,pca,bool_mask,n=4,var="so"):
    fig, ax = plt.subplots(n,2,figsize=(20,5*n))

    EOFs=pca.components_

    map_ = np.zeros((332, 362), dtype=float)
    map_[~bool_mask] = np.nan
    
    for i in range(n):
        map_[bool_mask] = EOFs[i]
        map_[~bool_mask] = np.nan
        map_ = 2 * map_ * dico["std"][var] + dico["mean"][var]
        
        im = ax[i,0].pcolor(map_/100000)
        fig.colorbar(im, ax=ax[i,0])

    for i in range(n):
        x=ax[i,1].plot(array[:,i],color="navy",alpha=0.9)  
        ax[i,1].axhline(0, c='red',linewidth="0.6")
        ax[i,1].grid('off')

    plt.tight_layout()
    plt.show()

    
def plot_all(dico,n=4,var="so"):
    fig, ax = plt.subplots(n,2,figsize=(20,5*n))

    EOFs=dico["pca"][var].components_

    map_ = np.zeros((332, 362), dtype=float)
    map_[~dico["mask"][var]] = np.nan
    
    for i in range(n):
        map_[dico["mask"][var]] = EOFs[i]
        map_[~dico["mask"][var]] = np.nan
        map_ = 2 * map_ * dico["std"][var] + dico["mean"][var]
        
        im = ax[i,0].pcolor(map_/100000)
        fig.colorbar(im, ax=ax[i,0])

    for i in range(n):
        x=ax[i,1].plot(dico["features"][var][:,i],color="navy",alpha=0.9)  
        ax[i,1].axhline(0, c='red',linewidth="0.6")
        ax[i,1].grid('off')

    plt.tight_layout()
    plt.show()
    
    
###############################
#  Mode/variance separation   #
#         method 4            #
############################### 
    
def extend_values(var,depth): #var.values #var.deptht.values
    result = [0]*round(depth[-1])
    for i in range(1,len(var)):
        b1 = round(depth[i-1])
        b2 = round(depth[i])
        result[b1:b2] = [var[i-1]]*(b2-b1)
    return np.array(result[1:])

def variance(point,serie):
    s1,s2 =serie[:point],serie[point:]
    v1 = np.sqrt(np.nansum((s1 - np.nanmean(s1)) ** 2))
    v2 = np.sqrt(np.nansum((s2 - np.nanmean(s2)) ** 2))
    return v1+v2

def min_var(serie,n_modes):
    variances = [variance(i,serie) for i in range(len(serie))]
    p = np.argmin(np.array(variances))
    if n_modes == 0:
        return (n_modes,p)
    else:
        v1,v2 = serie[:p],serie[p:]
        return (n_modes,p),min_var(v1,n_modes = n_modes-1), min_var(v2,n_modes = n_modes-1)
    

def mode1(sim,var,d,depth="olevel"):
    dataset = xr.open_dataset(sim,chunks={"time_counter": 100},decode_times=False)
    var1 = dataset[var].sel({depth: slice(None, d)}).mean(depth).load()
    var2 = dataset[var].sel({depth: slice(d, None)}).mean(depth).load()
    return var1.values,var2.values


def prepare(array,var="so"):
    array[array == 0.] = np.nan   
    array = array.astype(np.float32) 
    
    mean = np.nanmean(array)
    std = np.nanstd(array)
    array = (array - mean) / (2*std)  
    
    return array,mean,std