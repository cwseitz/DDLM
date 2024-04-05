import numpy as np
import torch

def errors2d(xy,xy_est):
    def match_on_xy(xy,xy_est,tol=5.0):
        dist = np.linalg.norm(xy_est-xy, axis=1)
        a = dist <= tol
        b = np.any(a)
        if b:
            idx = np.argmin(dist)
            c = np.squeeze(xy_est[idx])
            xerr,yerr = xy[0]-c[0],xy[1]-c[1]
        else:
            xerr,yerr = None,None
        return b,xerr,yerr

    nspots,_ = xy.shape
    all_bool = []; all_x_err = []; all_y_err = []
    
    for n in range(nspots):
        this_xy = xy[n,:]
        bool,xerr,yerr = match_on_xy(this_xy,xy_est)
        all_bool.append(bool)
        if xerr is not None and yerr is not None:
            all_x_err.append(xerr)
            all_y_err.append(yerr)   
                  
    all_bool = np.array(all_bool)
    all_x_err = np.array(all_x_err)
    all_y_err = np.array(all_y_err)
    return all_x_err, all_y_err
