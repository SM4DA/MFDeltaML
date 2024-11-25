import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Model_MFML import ModelMFML as MF

def KRR(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, sigma:float, reg: float, type_kernel:str='gaussian'):
    #generate the correct kernel matrix as prescribed by args parser
    import qml.kernels as k
    from qml.math import cho_solve
    if type_kernel=='matern':
        K_train = k.matern_kernel(X_train,X_train,sigma, order=1, metric='l2')
        K_test = k.matern_kernel(X_train,X_test,sigma, order=1, metric='l2')
    elif type_kernel=='laplacian':
        K_train = k.laplacian_kernel(X_train,X_train,sigma)
        K_test = k.laplacian_kernel(X_train,X_test,sigma)
    elif type_kernel=='gaussian':
        K_train = k.gaussian_kernel(X_train,X_train,sigma)
        K_test = k.gaussian_kernel(X_train,X_test,sigma)
    elif type_kernel=='linear':
        K_train = k.linear_kernel(X_train,X_train)
        K_test = k.linear_kernel(X_train,X_test)
    elif type_kernel=='sargan':
        K_train = k.sargan_kernel(X_train,X_train,sigma,gammas=None)
        K_test = k.sargan_kernel(X_train,X_test,sigma,gammas=None)
    
    #regularize 
    K_train[np.diag_indices_from(K_train)] += reg
    #train
    alphas = cho_solve(K_train,y_train)
    #predict
    preds = np.dot(alphas, K_test)
    #MAE calculation
    mae = np.mean(np.abs(preds-y_test))
    
    return mae

def SF_learning_curve(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, 
                      sigma:float=30, reg:float=1e-10, navg:int=10, ker:str='laplacian'):
    full_maes = np.zeros((11),dtype=float)
    for n in tqdm(range(navg), desc='avg loop for SF LC'):
        maes = []
        X_train,y_train = shuffle(X_train, y_train, random_state=42)
        for i in range(1,12):
            #start_time = time.time()
            temp = KRR(X_train[:2**i],X_test,y_train[:2**i],y_test,sigma=sigma,reg=reg,type_kernel=ker)
            maes.append(temp)
        full_maes += np.asarray(maes)
    
    full_maes = full_maes/navg
    return full_maes

def nested_same_hyperparams(X_train, energies, indexes, X_test, y_test, X_val, y_val, reg:str=1e-9, sig:float=200.0, ker:str='laplacian', navg:int=10):
    
    all_maes = np.zeros((11),dtype=float)
    all_olsmaes = np.zeros((11),dtype=float)
    nfids = energies.shape[0]
    
    
    for n in tqdm(range(navg),desc='avg-run loop for MFML and o-MFML LC...',leave=False):
        maes = []
        ols_maes = []
        for i in range(1,12):
            n_trains = np.asarray([2**(i+4),2**(i+3),2**(i+2),2**(i+1),2**(i)])[5-nfids:]
            
            #instantiate models
            model = MF(reg=reg, kernel=ker, sigma=sig,
                   order=1, metric='l2', gammas=None, 
                   p_bar=False)
            
            #train models
            model.train(X_train_parent=X_train, y_trains=energies, 
                    indexes=indexes, 
                    shuffle=True, n_trains=n_trains, 
                    seed=n)
            
            
            #default predictions
            _ = model.predict(X_test=X_test, X_val=X_val,
                              y_test=y_test, y_val=y_val, 
                              optimiser='default')
            maes.append(model.mae)
            
            #OLS predictions
            _ = model.predict(X_test=X_test, X_val=X_val,
                              y_test=y_test, y_val=y_val, 
                              optimiser='OLS')
            ols_maes.append(model.mae)
        
        #store MAEs into overall arrays
        all_maes[:] += np.asarray(maes)
        all_olsmaes[:] += np.asarray(ols_maes)
        
    
    return all_maes/navg, all_olsmaes/navg
    

def main(sig,reg=1e-10):
    indexes = np.load('raws/indexes.npy',allow_pickle=True)[1:]
    X_train = np.load(f'raws/X_train_CM.npy')
    energies = np.load(f'raws/energies.npy',allow_pickle=True)
    y_lower_train = energies[0]
    avg_lower = np.mean(y_lower_train)
    y_lower_train = y_lower_train-avg_lower
    energies = energies[1:]
    #centering energies
    for i in range(4):
        avg=np.mean(energies[i])
        try:
            energies[i] = energies[i] - avg - y_lower_train
        except:
            energies[i] = energies[i] - avg - y_lower_train[:2266]
    
    #test and val data
    X_test = np.load(f'raws/X_test_CM.npy')
    X_val = np.load(f'raws/X_val_CM.npy')
    y_test_lower = np.load(f'raws/y_lower_STO3G.npy') - avg_lower #center wrt train lower
    y_val_lower = np.load(f'raws/y_val_STO3G.npy') - avg_lower #center wrt train lower
    y_test = np.load(f'raws/y_test.npy') - avg - y_test_lower
    y_val = np.load(f'raws/y_val.npy') - avg - y_val_lower
    
    all_maes = np.zeros((4),dtype=object)
    def_maes = np.zeros((4),dtype=object)
    
    for fb in tqdm(range(3),desc=f'baseline loop'):
        def_maes[3-fb],all_maes[3-fb]= nested_same_hyperparams(X_train, energies[fb:], 
                                                               indexes[fb:], X_test, y_test, X_val, 
                                                               y_val, reg=reg, sig=sig, 
                                                               ker='matern', navg=10)
    
    np.save(f'outs/DPe_Delta_defaultMAEs_STO3G.npy',def_maes,allow_pickle=True)
    np.save(f'outs/DPe_Delta_OLSMAEs_STO3G.npy',all_maes,allow_pickle=True)
    

if __name__=='__main__':
    main(sig=3000.0,reg=1e-10)
    
