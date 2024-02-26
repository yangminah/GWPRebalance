"""
This script contains all the helper functions needed for the third data structure I used. 
The data structure consists of:
input_3d: [U, T]
input_loc: [sp, alphas]
target: [gU]
"""

from netCDF4 import Dataset
import numpy as np
import torch

from utils import get_la_lo_tstart

VAR_NAMES = ["input_3d", "target"]
M = 11796480
la, lo, tstart = get_la_lo_tstart(np.array(range(M)))
INDS = np.zeros((1440, 64, 128)).astype(int)
for i in range(M):
    INDS[tstart[i], la[i], lo[i]] = i
del tstart, la, lo, i


weightedLoss = True


# batch_alloc = [torch.zeros(bs,2,40), torch.zeros(bs), torch.zeros(bs,40)]
def get_batch(
    data, idxst, idxend, index_info, batch_alloc, transform=None, load2mem=True
):
    # print(f"transform={transform}, load2mem={load2mem}")
    bs = idxend - idxst
    # input_3d, SP, lat, lon, output_target, n_mem = data
    la, lo, tstart = index_info
    x3, xloc, target, pgU = batch_alloc
    xloc[:bs] = (
        1
        if isinstance(data["alphas"], (int, np.float64, float))
        else torch.from_numpy(
            data["alphas"][tstart[idxst:idxend], la[idxst:idxend], lo[idxst:idxend]]
        )
    )

    if load2mem is True:
        x3[:bs, :, :] = torch.from_numpy(
            data["input_3d"][
                tstart[idxst:idxend], :, la[idxst:idxend], lo[idxst:idxend], :
            ].reshape(bs, 2, 40)
        )
        target[:bs, :] = torch.from_numpy(
            data["target"][tstart[idxst:idxend], :, la[idxst:idxend], lo[idxst:idxend]]
        )
        return x3, xloc, target, pgU


def get_batch2(
    data, idxst, idxend, index_info, batch_alloc, transform=None, load2mem=True
):
    # print(f"transform={transform}, load2mem={load2mem}")
    bs = idxend - idxst
    # input_3d, SP, lat, lon, output_target, n_mem = data
    la, lo, tstart = index_info
    x3, xloc, target = batch_alloc
    xloc[:bs] = (
        1
        if isinstance(data["alphas"], (int, np.float64, float))
        else torch.from_numpy(
            data["alphas"][tstart[idxst:idxend], la[idxst:idxend], lo[idxst:idxend]]
        )
    )

    if load2mem is True:
        x3[:bs, :, :] = torch.from_numpy(
            data["input_3d"][
                tstart[idxst:idxend], :, la[idxst:idxend], lo[idxst:idxend], :
            ].reshape(bs, 2, 40)
        )
        target[:bs, :] = torch.from_numpy(
            data["target"][tstart[idxst:idxend], :, la[idxst:idxend], lo[idxst:idxend]]
        )
        return x3, xloc, target


def get_data(year, transform=None, load2mem=True, data={}, alphas=1):
    # fid = Dataset(f"/scratch/epg2/mima-output/mima2.0-NH-optimal_d{year*360:05}/atmos_4xdaily.nc","r",swmr=True)
    fid = Dataset(
        f"/scratch/projects/gerberlab/epg2/mima-output/mima2.0-NH-optimal_d{year*360:05}/atmos_4xdaily.nc",
        "r",
        swmr=True,
    )

    if load2mem is True:
        if len(data) == 0:
            data["input_3d"] = np.concatenate(
                (
                    np.array(fid["ucomp"][()]).reshape(1440, 40, 64, 128, 1),
                    np.array(fid["temp"][()]).reshape(1440, 40, 64, 128, 1),
                ),
                axis=-1,
            )
            data["alphas"] = np.sqrt(alphas)
            data["target"] = np.array(fid["gwfu_cgwd"][()])
            fid.close()
            data["sp"] = 1
            if transform:
                transform(data)
            return data
        else:  # upload in-place
            data["input_3d"][:, :, :, :, 0] = np.array(fid["ucomp"][()])
            data["input_3d"][:, :, :, :, 1] = np.array(fid["temp"][()])
            data["target"][:, :, :, :] = np.array(fid["gwfu_cgwd"][()])
            data["sp"] = 1
            fid.close()
            data["alphas"] = np.sqrt(alphas)
            if transform:
                transform(data)
            return data
    else:
        data["input_3d"] = [fid["ucomp"], fid["temp"]]
        data["alphas"] = np.sqrt(alphas)
        data["target"] = fid["gwfu_cgwd"]
        data["sp"] = 1
        return data


def get_data2(year, transform=None, load2mem=True, data={}, alphas=1):
    # fid = Dataset(f"/scratch/epg2/mima-output/mima2.0-NH-optimal_d{year*360:05}/atmos_4xdaily.nc","r",swmr=True)
    # fid = Dataset(f"/scratch/projects/gerberlab/epg2/mima-output/mima2.0-NH-optimal_d{year*360:05}/atmos_4xdaily.nc","r",swmr=True)
    fid = Dataset(
        f"/Users/minahyang/Documents/research/gwp/data/year{year:02d}atmos_4xdaily.nc",
        "r",
    )
    if load2mem is True:
        if len(data) == 0:
            data["input_3d"] = np.concatenate(
                (
                    np.array(fid["ucomp"][()]).reshape(1440, 40, 64, 128, 1),
                    np.array(fid["temp"][()]).reshape(1440, 40, 64, 128, 1),
                ),
                axis=-1,
            )
            data["alphas"] = np.sqrt(alphas)
            data["target"] = np.array(fid["gwfu_cgwd"][()])
            fid.close()
            data["sp"] = 1
            if transform:
                transform(data)
            return data
        else:  # upload in-place
            data["input_3d"][:, :, :, :, 0] = np.array(fid["ucomp"][()])
            data["input_3d"][:, :, :, :, 1] = np.array(fid["temp"][()])
            data["target"][:, :, :, :] = np.array(fid["gwfu_cgwd"][()])
            data["sp"] = 1
            fid.close()
            data["alphas"] = np.sqrt(alphas)
            if transform:
                transform(data)
            return data
    else:
        data["input_3d"] = [fid["ucomp"], fid["temp"]]
        data["alphas"] = np.sqrt(alphas)
        data["target"] = fid["gwfu_cgwd"]
        data["sp"] = 1
        return data
