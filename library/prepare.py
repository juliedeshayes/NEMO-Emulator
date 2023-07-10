import pandas as pd
import os
import numpy as np
import pickle
import xarray as xr
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
from joblib import Parallel, delayed, parallel_backend


##################################
##################################
##                              ##
##                              ##
##  PREPARE LIB FOR NOTEBOOKS:  ##
##       LOAD_SIMU_GRIDS        ##
##       PREPARE_DATASET        ##
##                              ##
##                              ##
##################################
##################################



#####################
#  LOAD SIMULATION  #
##################### 


#### 1.1 FIND ALL PATHS FOR A GRID ###

#get all files from a file ending (ex: grid_U.nc) and a parametrisation ex("JZ-06")
def getData(path,term,param):
    data = {}
    for i in range(len(term)):
        grid = []
        for file in sorted(os.listdir(path)):
            if file.endswith(term[i]) and param in file: #add param!=""
                grid.append(path+"/"+file)
        if(len(grid))>0:
            data[term[i]] = grid
    return data



######################
#                    #
#  EXTRACT FEATURES  #
#                    #
######################


#### LOAD XARRAY DATASET #####

#2.1 & 2.2 

def getfeatures(path,filter,yearly,time_dim):
    file   = xr.open_dataset(path, decode_times=False)
    delete = set(file.variables) - set(filter)
    file   = file.drop_vars(delete)
    if yearly:
        file = file.coarsen({time_dim: 12}).mean()   #TO CHANGE WITH TIME DIM
    return file

#2.3

def get4Dfeatures(path,filter,mean_dim,yearly,time_dim):
    file   = xr.open_dataset(path, chunks={time_dim: 100}, decode_times=False)
    delete = set(file.variables) - set(filter)
    file   = file.drop_vars(delete)
    #file   = file.mean(mean_dim,skipna=True)
    if yearly:
        file = file.coarsen({time_dim: 12}).mean().load() #TO CHANGE WITH TIME DIM
    return file

def getSSCA(ds,filter):
    ssca_dico = {}  
    if filter is not None:
        to_delete = set(ds.variables) - set(filter)
        ds      = ds.drop_vars(to_delete)
    for var in ds.variables:
        x, y    = ds[var].shape[1:3]
        nbyears = ds[var].shape[0] // 12 
        ssca    = np.reshape(np.array(ds[var]), (nbyears, 12, x, y))
        ssca    = np.mean(ssca, axis=0)
        ssca_dico[var] = ssca
    return ssca_dico

#### TRANSFORM TO DICO ####

def prepare(ds):
    features_dico,mean_dico,std_dico = {},{},{}
    for var in ds.variables:
        ds[var].values[ds[var].values == 0.] = np.nan   
        ds[var] = ds[var].astype(np.float32) 
        
        mean    = ds.mean()
        std     = ds.std()
        ds[var] = (ds[var] - mean[var]) / (2*std[var])  
        
        features_dico[var] = ds[var].values
        mean_dico[var]     = mean[var].values
        std_dico[var]      = std[var].values
    return features_dico, mean_dico, std_dico  


def toDictionnary(grid,dtime,nb_dim,filter=None,yearly=None,mean_dim=None,cut_simu=0,n_jobs=10):
    if nb_dim == 4:
        ds = list(Parallel(n_jobs)(delayed(get4Dfeatures)(path,filter,mean_dim,yearly,dtime) for path in grid))
    else:
        ds = list(Parallel(n_jobs)(delayed(getfeatures)(path,filter,yearly,dtime) for path in grid))
    ds   = xr.concat(ds, dtime)                   
    ds   = ds.sel({dtime: slice(cut_simu, None)})           #TO CHANGE WITH TIME DIM
    ssca = getSSCA(xr.open_dataset(grid[-1], decode_times=False),filter)
    features,mean,std = prepare(ds)
    return {"features":features,"mean":mean,"std":std,"ssca":ssca}  


###############################
#                             #
#  TTRANSFORM TO TIME SERIES  #
#                             #
###############################


def get_nan_values(array): 
    bool_mask = np.asarray(np.isfinite(array), dtype=bool)
    return bool_mask 

def apply_nan_mask(simu):
    mask_dico={}
    for var in simu["features"].keys():
        mask = get_nan_values(simu["features"][var][0])
        simu["features"][var] = simu["features"][var][:,mask]
        mask_dico[var] = mask
    simu["mask"] = mask_dico
    return simu

def toTimeSeries(simu,n_comp):
    simu      = apply_nan_mask(simu)
    pca_dico  = {}
    
    for var in simu["features"].keys():
        pca = PCA(n_comp, whiten=False)
        simu["features"][var] = pca.fit_transform(simu["features"][var])
        pca_dico[var] = pca
    simu["pca"] = pca_dico
    return simu


###############################
#                             #
#  CONCATENATE DICTIONNARIES  #
#                             #
###############################


#### 4.1 CONCATENATE FEATURE FROME THE SAME GRID ####


def concatenateDico(dicos):
    grid_dico = {"features":{},"mean":{},"std":{},"ssca":{},"mask":{},"pca":{}}
    for dico in dicos:
        for info in dico.keys():
            for var in dico[info].keys():
                grid_dico[info][var] = dico[info][var]
    return grid_dico


#### 4.2 CONCATENATE FEATURE FROME THE SAME SIMULATION ####

def concatenateGrid(grids):
    simu_dico = None
    for grid in grids:
        if simu_dico == None:
            simu_dico = grid
        else:
            for info in simu_dico.keys():
                for var in simu_dico[info].keys():
                    simu_dico[info][var] = grid[info][var]
    return simu_dico


#################################
#                               #
#  SAVE SIMULATION DICTIONNARY  #
#                               #
#################################

def toList(dico):
    del dico['pca']
    for info in dico.keys():
        for var in dico[info].keys():
            dico[info][var] = dico[info][var].tolist()
    return dico

def saveSimu(dico,yearly,n):
    simulation_pca  = dico["pca"]
    simulation_dico = toList(dico)
    
    if yearly: 
        time_axis = "yearly"
    else:
        time_axis = "monthly"
        
    #save the simu info
    np.savez(f"../datasets/{time_axis}/simu/simu{n}_dico", **simulation_dico)
    #save the pca info
    with open(f"../datasets/{time_axis}/pca/simu{n}_pca", 'wb') as file:
        pickle.dump(simulation_pca, file)  
    print(f"you file where successfully saved in the dataset/{time_axis}/pca and the dataset/{time_axis}/simu folders")
    print(f"the number of the simulation is {n}")

    
#################################
#                               #
#  LOAD SIMULATION DICTIONNARY  #
#                               #
#################################

#1.1

def loadSimu(n,yearly):
    if yearly: 
        time_axis = "yearly"
    else:
        time_axis = "monthly"
    file = f"/gpfswork/rech/omr/uen17sn/NewSpinUp/datasets/{time_axis}/simu/simu{n}_dico.npz"
    #file = f"../datasets/{time_axis}/simu/simu{n}_dico.npz"
    simu_dico = np.load(file,allow_pickle=True)
    simu_dico = {key: simu_dico[key] for key in simu_dico.keys()}
    for var in simu_dico: simu_dico[var] = simu_dico[var].item()
    return simu_dico

def createDataFrame(dico,main):
    df = None
    for feature in dico["features"].keys():
        array = np.array(dico["features"][feature])
        names = [f"{feature}-{i+1}" for i in range(np.shape(array)[1])]
        if feature == main:
            names = [main]
            array =  np.array(dico["features"][feature])[:,0]
        if df is None:
            df = pd.DataFrame(array,columns=names)
        else:
            df[names]= array
    return df

def createDataset(simu,yearly,main="MSFT",filter=None):
    simu_dico = loadSimu(simu,yearly)
    df        = createDataFrame(simu_dico,main)
    
    if filter is not None:
        if set(filter).issubset(set(simu_dico["features"].keys())):
            delete = set(simu_dico["features"].keys()) - set(filter) 
            df     = df.drop(columns=delete) 
        else:
            print("File doesn't contains required features")
            return 0        
    return df