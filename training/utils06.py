"""
This script contains all the helper functions needed for the third data structure I used. 
The data structure consists of:
input_3d: [U,V,W,T]
input_loc: [lat lon ps]
target: [gU]
"""
from netCDF4 import Dataset
import numpy as np
import torch

from utils import get_la_lo_tstart
# VAR_NAMES = ['input_3d', 'sp', 'target']
# M = 11796480
# la,lo,tstart=get_la_lo_tstart(np.array(range(M)))
# INDS = np.zeros((1440,64,128)).astype(int)
# for i in range(M):
#     INDS[tstart[i],la[i],lo[i]]=i
# del tstart, la, lo, i


def get_batch(data, idxst, idxend, index_info, batch_alloc, transform=None, load2mem=True):
    #print(f"transform={transform}, load2mem={load2mem}")
    bs = idxend-idxst
    #input_3d, SP, lat, lon, output_target, n_mem = data
    la, lo, tstart = index_info
    x3, xloc, target = batch_alloc
    lat = data['lat']
    lon = data['lon']

    xloc[:bs,0] = torch.from_numpy(lat[la[idxst:idxend]])
    xloc[:bs,1] = torch.from_numpy(lon[lo[idxst:idxend]])
    if load2mem==True:
        # x3[:bs,:,:] = torch.from_numpy(data['input_3d'][tstart[idxst:idxend],:,la[idxst:idxend],lo[idxst:idxend],:].reshape(bs,4,40))
        x3[:bs,:,:] = torch.transpose(torch.from_numpy(data['input_3d'][tstart[idxst:idxend],:,la[idxst:idxend],lo[idxst:idxend],:]),1,2)
        xloc[:bs,2] = torch.from_numpy(data['sp'][tstart[idxst:idxend],la[idxst:idxend],lo[idxst:idxend]])
        target[:bs,:] = torch.from_numpy(data['target'][tstart[idxst:idxend],:,la[idxst:idxend],lo[idxst:idxend]])
        return x3, xloc, target
    # else:
    #     for b in range(idxst, idxend):
    #         #print(f"sample: {b}")
    #         for i in range(2):
    #             target[b-idxst,:,:,i] = torch.from_numpy(output_target[i][tstart[b],:,data['la'][b],data['lo'][b]])
    #         for i in range(4):
    #             x3[b-idxst,t,:,i] = torch.from_numpy(input_3d[i][tstart[b]-tt,:,data['la'][b],data['lo'][b]])
    #             xloc[b-idxst,2+t] = torch.from_numpy(SP[tstart[b]-tt,data['la'][b],data['lo'][b]])
    #     if transform:
    #         x3, xloc, target = transform([x3, xloc, target])
    #     return x3, xloc, target

def get_data(year, transform=None, load2mem=True, data={}):
    fid = Dataset(f"/scratch/projects/gerberlab/epg2/mima-output/mima2.0-NH-optimal_d{year*360:05}/atmos_4xdaily.nc","r")
    # fid = Dataset(f"/scratch/epg2/mima-output/mima2.0-NH-optimal_d{year*360:05}/atmos_4xdaily.nc","r")
    if load2mem==True:
        if len(data) == 0:
            data['input_3d'] = np.concatenate((
                np.array(fid["ucomp"][()]).reshape(1440,40,64,128,1),
                np.array(fid["vcomp"][()]).reshape(1440,40,64,128,1),
                np.array(fid["omega"][()]).reshape(1440,40,64,128,1),
                np.array(fid["temp"][()]).reshape(1440,40,64,128,1)
                ),axis=-1
            )
            data['sp']=fid["ps"][()]/100
            data['lat']=np.array(fid["lat"][()])
            data['lon']=np.array(fid["lon"][()])
            data['target'] = np.array(fid["gwfu_cgwd"][()])
            fid.close()
            if transform:
                transform(data)
            return data
        else: # upload in-place
            data['input_3d'][:,:,:,:,0]=np.array(fid["ucomp"][()])
            data['input_3d'][:,:,:,:,1]=np.array(fid["vcomp"][()])
            data['input_3d'][:,:,:,:,2]=np.array(fid["omega"][()])
            data['input_3d'][:,:,:,:,3]=np.array(fid["temp"][()])
            data['sp'][:,:,:]=fid['ps'][()]/100
            data['target'][:,:,:,:]=np.array(fid["gwfu_cgwd"][()])
            fid.close()
            if transform:
                transform(data)
            return data
    else:
        data['input_3d'] =[fid["ucomp"],fid["vcomp"],fid["omega"],fid["temp"]]
        data['sp']=fid["ps"]
        data['lat']=np.array(fid["lat"][()])
        data['lon']=np.array(fid["lon"][()])
        data['target'] = fid["gwfu_cgwd"]
        return data
