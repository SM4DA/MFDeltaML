import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import qml.kernels as k
from qml.math import cho_solve

def KRR(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, sigma:float, reg: float, type_kernel:str='gaussian'):
    #generate the correct kernel matrix as prescribed by args parser
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
    nmax = 11
    
    full_maes = np.zeros((nmax),dtype=float)
    for n in tqdm(range(navg), desc='avg loop for SF LC',leave=True):
        maes = []
        X_train,y_train = shuffle(X_train, y_train, random_state=42)
        for i in tqdm(range(1,nmax+1),desc='loop over training sizes',leave=False):
            #start_time = time.time()
            temp = KRR(X_train[:2**i],X_test,y_train[:2**i],y_test,sigma=sigma,reg=reg,type_kernel=ker)
            maes.append(temp)
        full_maes += np.asarray(maes)
    
    full_maes = full_maes/navg
    return full_maes

def main(sig,reg):
    X_train = np.load('raws/X_train_CM.npy')
    X_test = np.load('raws/X_test_CM.npy')
    y_trains = np.load('raws/energies.npy',allow_pickle=True)
    
    y_upper = y_trains[-1]
    avg_upper = np.mean(y_upper)
    y_upper = y_upper-avg_upper
    
    fids = ['STO3G','321G','631G','SVP']
    
    for i in range(4):
        #lower fidelity
        y_lower = y_trains[i]
        avg_lower = np.mean(y_lower)
        y_lower = y_lower-avg_lower #centering
        
        del_y_train = y_upper-y_lower #centered delta value
    
        #test energies - centered
        y_test_upper = np.load('raws/y_test.npy') - avg_upper
        y_test_lower = np.load(f'raws/y_lower_{fids[i]}.npy') - avg_lower
        
        del_y_test = y_test_upper-y_test_lower
    
        maes = SF_learning_curve(X_train=X_train, X_test=X_test, y_train=del_y_train, y_test=del_y_test, 
                                 sigma=sig, reg=reg, navg=10, ker='matern')
        np.save(f'outs/delML_{fids[i]}_mae.npy',maes)
    
    
if __name__=='__main__':
    main(sig=150.0,reg=1e-10)