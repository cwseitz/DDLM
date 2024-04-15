import numpy as np
import torch

def errors2d(xy,xy_est):
    def match_on_xy(xy_est,xy,tol=5.0):
        dist = np.linalg.norm(xy-xy_est, axis=1)
        a = dist <= tol
        b = np.any(a)
        if b:
            idx = np.argmin(dist)
            c = np.squeeze(xy[idx])
            xerr,yerr = xy_est[0]-c[0],xy_est[1]-c[1]
        else:
            xerr,yerr = None,None
        return b,xerr,yerr

    num_found,_ = xy_est.shape
    nspots,_ = xy.shape
    bfound = []; all_x_err = []; all_y_err = []
    
    for n in range(num_found):
        this_xy_est = xy_est[n,:]
        bool,xerr,yerr = match_on_xy(this_xy_est,xy)
        bfound.append(bool)
        if xerr is not None and yerr is not None:
            all_x_err.append(xerr)
            all_y_err.append(yerr)   
                  
    inter = np.sum(np.array(bfound).astype(int)) #intersection
    fp = len(bfound)-sum(bfound)
    union = nspots+fp
    fn = nspots - inter
    all_x_err = np.array(all_x_err)
    all_y_err = np.array(all_y_err)
    return all_x_err, all_y_err, inter, union, fp, fn
