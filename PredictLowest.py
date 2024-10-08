import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from qml.math import cho_solve
import qml.kernels as k

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
    
    return mae, preds



def main(sig, reg):
    X_train = np.load('raws/X_train_CM.npy')
    X_test = np.load('raws/X_test_CM.npy')
    y_trains = np.load('raws/energies.npy',allow_pickle=True)
    
    y_upper_train = y_trains[-1]
    avg_upper = np.mean(y_upper_train)
    y_upper_train = y_upper_train-avg_upper
    #only for sto3g
    y_lower_train = y_trains[0]
    avg_lower = np.mean(y_lower_train)
    y_lower_train = y_lower_train-avg_lower
    
    y_train_delta = y_upper_train - y_lower_train
    
    #test energies - centered
    y_test_upper = np.load('raws/y_test.npy') - avg_upper
    
    full_maes = np.zeros((nmax),dtype=float)
    baseline_MAE = np.zeros((nmax),dtype=float)
    
    ref_y_lower = np.load('raws/y_lower_STO3G.npy')-avg_lower
    
    for n in tqdm(range(navg), desc='avg loop for SF LC',leave=True):
        maes = []
        base_mae = []
        X_train, y_train_delta, y_lower_train = shuffle(X_train, y_train_delta, y_lower_train, random_state=42)
        for i in tqdm(range(1,nmax+1),desc='loop over training sizes',leave=False):            
            #make preds - these are centered predictions of baseline 
            b_m, y_test_lower = KRR(X_train = X_train[:2**i], X_test = X_test, 
                                    y_train = y_lower_train[:2**i], y_test = ref_y_lower,
                                    sigma=sig, reg=reg, type_kernel='matern')
            base_mae.append(b_m)
            #prepare Delta of prediction energies for MAE
            del_y_test = y_test_upper - y_test_lower
            
            temp,_ = KRR(X_train = X_train[:2**i], X_test = X_test, 
                       y_train = y_train_delta[:2**i], y_test = del_y_test, 
                       sigma=sig, reg=reg, type_kernel='matern')
            maes.append(temp)
        full_maes += np.asarray(maes)
        baseline_MAE += np.asarray(base_mae)
    
    full_maes = full_maes/navg
    baseline_MAE = baseline_MAE/navg
    
    #return full_maes, baseline_MAE
    np.save(f'outs/predictedbase_delML_STO3G_mae.npy',full_maes)
    np.save(f'outs/predictedbase_delML_baseline_mae.npy',baseline_MAE)

if __name__=='__main__':
    nmax = 11
    navg=10
    main(sig=150.0,reg=1e-10)
        
        