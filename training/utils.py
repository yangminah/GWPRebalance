import os
from time import perf_counter
from logging import info, warning

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import h5py
from netCDF4 import Dataset

VAR_NAMES = ["input_3d", "sp", "target"]
M = 11796480
la, lo, tstart = get_la_lo_tstart(np.array(range(M)))
INDS = np.zeros((1440, 64, 128)).astype(int)

def add_l1_reg(model, l1_amp):
    l1_reg = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if "weight" in name:
            l1_reg = l1_reg + torch.norm(param, 1)
    return l1_reg


def compute_errs(data, stats, batch_info, model, errors, save_type):
    model.eval()
    stats0, stats1 = stats
    (
        lastbatch_va,
        bs,
        N_va,
        get_batch,
        [la, lo, tstart],
        batch_alloc,
        transform_cc,
        load2mem,
        device,
    ) = batch_info
    index_info_va = [la, lo, tstart]
    evecs = {} if save_type == dict else h5py.File(save_type, "a", swmr=True)
    for error in errors:
        if error == "absolute_component_errors" and error not in evecs.keys():
            evecs[error] = np.zeros((M, 40))
        elif error == "relative_component_errors" and error not in evecs.keys():
            evecs[error] = np.zeros((M, 40))
        elif error == "standardized_component_errors" and error not in evecs.keys():
            evecs[error] = np.zeros((M, 40))
        elif error == "relative_spread_errors" and error not in evecs.keys():
            evecs[error] = np.zeros((M, 40))
        elif error == "absolute_norm_errors" and error not in evecs.keys():
            evecs[error] = np.zeros(M)
        elif error == "2norms" and error not in evecs.keys():
            evecs[error] = np.zeros(M)
        elif error == "relative_norm_errors" and error not in evecs.keys():
            evecs[error] = np.zeros(M)
    for i in range(lastbatch_va):
        idxst = i * bs
        idxend = M if i == lastbatch_va - 1 else (i + 1) * bs
        bs2 = idxend - idxst
        with torch.no_grad():
            x3, xloc, gU = get_batch(data, idxst, idxend, index_info_va, batch_alloc)
            x3, xloc, gU = x3.to(device), xloc.to(device), gU.to(device)
            pred_gU = model([x3, xloc])

            # * x3, xloc, gU, pgU = get_batch(data, idxst, idxend, index_info_va, batch_alloc)
            # * x3, xloc, gU, pgU = x3.to(device), xloc.to(device), gU.to(device), pgU.to(device)
            # * pred_gU = model([x3[:bs],pgU[:bs]])

        # Convert to numpy arrays
        pred_gU = torch.Tensor.cpu(pred_gU).detach().numpy()
        gU = torch.Tensor.cpu(gU).detach().numpy()
        upred_gU = (pred_gU * stats0["target_std"]) + stats0["target_mean"]
        ugU = (gU * stats0["target_std"]) + stats0["target_mean"]
        for error in errors:
            if error == "absolute_component_errors":
                evecs[error][idxst:idxend, :] = (upred_gU[:bs2] - ugU[:bs2]).reshape(
                    bs2, 40
                )
            elif error == "relative_component_errors":
                if "absolute_component_errors" in errors:
                    evecs[error][idxst:idxend, :] = evecs["absolute_component_errors"][
                        idxst:idxend
                    ] / (ugU[:bs2]).reshape(bs2, 40)
                else:
                    evecs[error][idxst:idxend, :] = (upred_gU[:bs2] - ugU[:bs2]) / (
                        (ugU[:bs2]).reshape(bs2, 40)
                    )
            elif error == "standardized_component_errors":
                if "absolute_component_errors" in errors:
                    evecs[error][idxst:idxend, :] = (
                        evecs["absolute_component_errors"][idxst:idxend]
                        / stats1["target_std"]
                    )
                else:
                    evecs[error][idxst:idxend, :] = (
                        upred_gU[:bs2] - ugU[:bs2]
                    ) / stats1["target_std"]
            elif error == "absolute_norm_errors":
                if "absolute_component_errors" in errors:
                    evecs[error][idxst:idxend] = np.linalg.norm(
                        evecs["absolute_component_errors"][idxst:idxend, :], axis=1
                    )
                else:
                    evecs[error][idxst:idxend, :] = np.linalg.norm(
                        (upred_gU[:bs2] - ugU[:bs2]).reshape(bs2, 40), axis=1
                    )
            elif error == "2norms":
                evecs[error][idxst:idxend] = np.linalg.norm(ugU[:bs2], axis=1)
            elif error == "relative_norm_errors":
                if "absolute_norm_errors" in errors:
                    if "2norms" in errors:
                        evecs[error][idxst:idxend] = (
                            evecs["absolute_norm_errors"][idxst:idxend]
                            / evecs["2norms"][idxst:idxend]
                        )
                    else:
                        evecs[error][idxst:idxend] = evecs["absolute_norm_errors"][
                            idxst:idxend
                        ] / np.linalg.norm(ugU[:bs2], axis=1)
                elif "absolute_component_errors" in errors:
                    if "2norms" in errors:
                        evecs[error][idxst:idxend] = (
                            np.linalg.norm(
                                evecs["absolute_component_errors"][idxst:idxend, :],
                                axis=1,
                            )
                            / evecs["2norms"][idxst:idxend]
                        )
                    else:
                        evecs[error][idxst:idxend] = np.linalg.norm(
                            evecs["absolute_component_errors"][idxst:idxend, :], axis=1
                        ) / np.linalg.norm(ugU[:bs2], axis=1)
                else:
                    if "2norms" in errors:
                        evecs[error][idxst:idxend] = (
                            np.linalg.norm(
                                (upred_gU[:bs2] - ugU[:bs2]).reshape(bs2, 40), axis=1
                            )
                            / evecs["2norms"][idxst:idxend]
                        )
                    else:
                        evecs[error][idxst:idxend] = np.linalg.norm(
                            (upred_gU[:bs2] - ugU[:bs2]).reshape(bs2, 40), axis=1
                        ) / np.linalg.norm(ugU[:bs2], axis=1)
    if isinstance(evecs, dict):
        return evecs
    else:
        evecs.close()
        return save_type


def compute_errs2(data, stats, batch_info, model, errors, save_type):
    model.eval()
    stats0, stats1 = stats
    (
        lastbatch_va,
        bs,
        N_va,
        get_batch,
        [la, lo, tstart],
        batch_alloc,
        transform_cc,
        load2mem,
        device,
    ) = batch_info
    index_info_va = [la, lo, tstart]
    evecs = {} if save_type == dict else h5py.File(save_type, "a", swmr=True)
    for error in errors:
        if error == "absolute_component_errors" and error not in evecs.keys():
            evecs[error] = np.zeros((M, 40))
        elif error == "relative_component_errors" and error not in evecs.keys():
            evecs[error] = np.zeros((M, 40))
        elif error == "standardized_component_errors" and error not in evecs.keys():
            evecs[error] = np.zeros((M, 40))
        elif error == "relative_spread_errors" and error not in evecs.keys():
            evecs[error] = np.zeros((M, 40))
        elif error == "absolute_norm_errors" and error not in evecs.keys():
            evecs[error] = np.zeros(M)
        elif error == "2norms" and error not in evecs.keys():
            evecs[error] = np.zeros(M)
        elif error == "relative_norm_errors" and error not in evecs.keys():
            evecs[error] = np.zeros(M)
    for i in range(lastbatch_va):
        idxst = i * bs
        idxend = M if i == lastbatch_va - 1 else (i + 1) * bs
        bs2 = idxend - idxst
        with torch.no_grad():
            # * x3, xloc, gU = get_batch(data, idxst, idxend, index_info_va, batch_alloc)
            # * x3, xloc, gU = x3.to(device), xloc.to(device), gU.to(device)
            # * pred_gU = model([x3,xloc])

            x3, xloc, gU, pgU = get_batch(
                data, idxst, idxend, index_info_va, batch_alloc
            )
            x3, xloc, gU, pgU = (
                x3.to(device),
                xloc.to(device),
                gU.to(device),
                pgU.to(device),
            )
            pred_gU = model([x3[:bs], pgU[:bs]])

        # Convert to numpy arrays
        pred_gU = torch.Tensor.cpu(pred_gU).detach().numpy()
        gU = torch.Tensor.cpu(gU).detach().numpy()
        upred_gU = (pred_gU * stats0["target_std"]) + stats0["target_mean"]
        ugU = (gU * stats0["target_std"]) + stats0["target_mean"]
        for error in errors:
            if error == "absolute_component_errors":
                evecs[error][idxst:idxend, :] = (upred_gU[:bs2] - ugU[:bs2]).reshape(
                    bs2, 40
                )
            elif error == "relative_component_errors":
                if "absolute_component_errors" in errors:
                    evecs[error][idxst:idxend, :] = evecs["absolute_component_errors"][
                        idxst:idxend
                    ] / (ugU[:bs2]).reshape(bs2, 40)
                else:
                    evecs[error][idxst:idxend, :] = (upred_gU[:bs2] - ugU[:bs2]) / (
                        (ugU[:bs2]).reshape(bs2, 40)
                    )
            elif error == "standardized_component_errors":
                if "absolute_component_errors" in errors:
                    evecs[error][idxst:idxend, :] = (
                        evecs["absolute_component_errors"][idxst:idxend]
                        / stats1["target_std"]
                    )
                else:
                    evecs[error][idxst:idxend, :] = (
                        upred_gU[:bs2] - ugU[:bs2]
                    ) / stats1["target_std"]
            elif error == "absolute_norm_errors":
                if "absolute_component_errors" in errors:
                    evecs[error][idxst:idxend] = np.linalg.norm(
                        evecs["absolute_component_errors"][idxst:idxend, :], axis=1
                    )
                else:
                    evecs[error][idxst:idxend, :] = np.linalg.norm(
                        (upred_gU[:bs2] - ugU[:bs2]).reshape(bs2, 40), axis=1
                    )
            elif error == "2norms":
                evecs[error][idxst:idxend] = np.linalg.norm(ugU[:bs2], axis=1)
            elif error == "relative_norm_errors":
                if "absolute_norm_errors" in errors:
                    if "2norms" in errors:
                        evecs[error][idxst:idxend] = (
                            evecs["absolute_norm_errors"][idxst:idxend]
                            / evecs["2norms"][idxst:idxend]
                        )
                    else:
                        evecs[error][idxst:idxend] = evecs["absolute_norm_errors"][
                            idxst:idxend
                        ] / np.linalg.norm(ugU[:bs2], axis=1)
                elif "absolute_component_errors" in errors:
                    if "2norms" in errors:
                        evecs[error][idxst:idxend] = (
                            np.linalg.norm(
                                evecs["absolute_component_errors"][idxst:idxend, :],
                                axis=1,
                            )
                            / evecs["2norms"][idxst:idxend]
                        )
                    else:
                        evecs[error][idxst:idxend] = np.linalg.norm(
                            evecs["absolute_component_errors"][idxst:idxend, :], axis=1
                        ) / np.linalg.norm(ugU[:bs2], axis=1)
                else:
                    if "2norms" in errors:
                        evecs[error][idxst:idxend] = (
                            np.linalg.norm(
                                (upred_gU[:bs2] - ugU[:bs2]).reshape(bs2, 40), axis=1
                            )
                            / evecs["2norms"][idxst:idxend]
                        )
                    else:
                        evecs[error][idxst:idxend] = np.linalg.norm(
                            (upred_gU[:bs2] - ugU[:bs2]).reshape(bs2, 40), axis=1
                        ) / np.linalg.norm(ugU[:bs2], axis=1)
    if isinstance(evecs, dict):
        return evecs
    else:
        evecs.close()
        return save_type


def count_params(model):
    num_params = 0
    for parameter in model.parameters():
        num_params += np.prod(parameter.shape[:])
    return num_params


def get_annualcycle(years, vn):
    dict = {"gU": "gwfu_cgwd", "gV": "gwfv_cgwd"}
    varname = dict[vn]
    for y, yr in enumerate(years):
        if y == 0:
            fid = load_year(yr)
            x = fid[varname][()]
            fid.close()
        else:
            fid = load_year(yr)
            x += fid[varname][()]
            fid.close()
    x /= len(years)
    return x


def get_la_lo_tstart(idx, n_mem=1, n_latlon=64 * 128):
    latlon = idx % n_latlon
    lo = latlon % 128
    la = latlon // 128
    tstart = idx // n_latlon
    tstart += n_mem - 1
    if type(idx) == np.ndarray:
        return la.astype(np.int32), lo.astype(np.int32), tstart.astype(np.int32)
    else:
        return la, lo, tstart



for i in range(M):
    INDS[tstart[i], la[i], lo[i]] = i
del tstart, la, lo, i


def get_latlon_tstart(idx, n_latlon=64 * 128):
    latlon = idx % n_latlon
    tstart = idx // n_latlon
    return latlon, tstart


def get_N(*, n_mem, portion=1):
    n_latlon = 64 * 128
    n_window = 1440 - n_mem + 1
    N_tr = n_latlon * n_window
    N_va = N_tr // portion
    return N_tr, N_va


def DB_get_bins_ind_alpha(
    functional, year, nbins, t, maxrepeat=5, returnalpha=False, returninds=False
):
    if functional == "urange":
        h5str = f"/scratch/lmy7879/data2analyze/urange.h5"
        fid = h5py.File(h5str, "r", swmr=True)
        var = fid[f"{year:02d}"][:, 0, :, :]
        fid.close()
    # Get original histogram
    # if len(var.shape)==4:
    #     var = var[:,0,:,:]
    bin_edges = np.linspace(np.min(var), np.max(var), num=nbins + 1)
    counts = np.zeros_like(bin_edges).astype(int)
    i = np.searchsorted(bin_edges[1:], var)
    np.add.at(counts, i, 1)
    with np.errstate(divide="ignore"):
        alphas = 1 - (t * (1 - M / (nbins * counts)))
    alphas[alphas == np.inf] = 0
    info(f"returnalpha is {returnalpha}.")
    if returnalpha is True:
        alphas[alphas > maxrepeat] = maxrepeat
        return alphas[i]
    else:
        global ind_list
        ind_list = []
        ind_lists = [] if returninds is True else 0
        for aa, alpha in enumerate(alphas):
            if alpha != np.inf:
                ind_i_aa = INDS[i == aa]
                if returninds is True:
                    ind_lists.append(ind_list)
                if alpha < 1:
                    ind_list.extend(
                        ind_i_aa[
                            np.random.choice(
                                counts[aa],
                                np.ceil(alpha * counts[aa]).astype(int),
                                replace=False,
                            )
                        ]
                    )
                elif alpha >= 1 and alpha < maxrepeat:
                    ind_list.extend(ind_i_aa.tolist() * np.floor(alpha).astype(int))
                    # for j in range(np.floor(alpha).astype(int)):
                    #     ind_list.extend(ind_i_aa)
                    ind_list.extend(
                        ind_i_aa[
                            np.random.choice(
                                counts[aa],
                                np.ceil((alpha - np.floor(alpha)) * counts[aa]).astype(
                                    int
                                ),
                                replace=False,
                            )
                        ]
                    )
                else:
                    ind_list.extend(INDS[i == aa].tolist() * maxrepeat)
                    # for j in range(maxrepeat):
                    #     ind_list.extend(INDS[i==aa])
        ind_array = np.asarray(ind_list)
        del ind_list
        M2 = ind_array.shape[0]
        if M2 < M:
            np.random.shuffle(ind_array)
        else:
            ind_array = ind_array[np.random.choice(M2, M, replace=False)]
        if returninds is True:
            return ind_array, ind_lists
        else:
            return ind_array


def get_padding(n_in, n_out, kernel, stride=1, dilation=1, conv=True):
    if conv is True:
        total_padding = (n_in - 1) * stride + dilation * (kernel - 1) + 1 - n_out
        if total_padding < 0:
            warning("Total padding is negative, something is wrong.")
        if total_padding % 2 == 1:
            return total_padding // 2, 1
        else:
            return total_padding // 2, 0


def get_stats_oneyear(h5str, stdim, year, verify):
    if stdim == 0:
        waxis = (0, 1, 2, 3)
    elif stdim == 1:
        waxis = (0, 2, 3)
    spaxis = (0, 1, 2)
    stats_fid = h5py.File(h5str, "a")
    dataloc = f"{year:02d}/stdim{stdim:d}"
    stats = {}
    try:
        stats_fid[dataloc]
        if verify is True:
            return verify
        else:
            for stat in ["mean", "std"]:
                for var in VAR_NAMES:
                    stats[f"{var}_{stat}"] = stats_fid[f"{dataloc}/{var}/{stat}"][()]
            stats_fid.close()
            return stats
    except KeyError:
        data = get_data(year)
        for var in VAR_NAMES:
            if var == "input_3d" or var == "target":
                axis = waxis
            else:
                axis = spaxis
            m, s = get_stats(data[var], axis=axis)
            stats_fid[f"{dataloc}/{var}/mean"] = m
            stats_fid[f"{dataloc}/{var}/std"] = s
            stats[f"{var}_mean"] = m
            stats[f"{var}_std"] = s
        stats_fid.close()
        del data
        if verify is True:
            return True
        else:
            return stats


# def get_stats(data,axis):
#     xbar = np(data.astype(np.double),axis=axis)
#     xstd = np.sqrt(np.mean((data.astype(np.double)-xbar)**2,axis=axis))
#     return xbar.astype(np.single), xstd.astype(np.single)


def get_tlvars(year):
    stats_fid = h5py.File("/home/lmy7879/gwp/h5files/trn_stats.h5", "r", swmr=True)
    stdim0stats = stats_fid[f"{year:02d}/stdim0/target"][()]
    fid = load_year(year)
    gU = (fid["gwfu_cgwd"][()] - stdim0stats[0, 0]) / stdim0stats[1, 0]
    gV = (fid["gwfv_cgwd"][()] - stdim0stats[0, 1]) / stdim0stats[1, 1]
    tlvars = np.zeros((40, 2))
    tlvars[:, 0] = np.var(gU, axis=(0, 2, 3))
    tlvars[:, 1] = np.var(gV, axis=(0, 2, 3))
    return tlvars


def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)


def load_year(year):
    return Dataset(
        f"/scratch/projects/gerberlab/epg2/mima-output/mima2.0-NH-optimal_d{year*360:05}/atmos_4xdaily.nc",
        "r",
    )
    # return Dataset(f"/scratch/epg2/mima-output/mima2.0-NH-optimal_d{year*360:05}/atmos_4xdaily.nc","r")

    # return Dataset(f"/scratch/os584/shared/mima-output/mima2.0-nh-optimal-master-cionni2011-distribute02000/mima2.0-nh-optimal-master-cionni2011-distribute02000_d{year*360:05}/atmos_4xdaily_d{year*360:05}.nc","r")


def load_year2(year):
    return Dataset(
        "/Users/minahyang/Documents/research/gwp/data/year60atmos_4xdaily.nc", "r"
    )


# def get_la_lo(latlon):
#     lo = latlon % 128
#     la = latlon // 128
#     return la, lo


def save_ao_get_urange(h5str, year):
    # Get source levels by latitude
    source_level_pressure = 315
    nc = load_year(21)
    lat = nc["lat"][()] / 180 * np.pi
    pfull = nc["pfull"][()]
    nc.close()
    klevel_of_source = 0
    for k, pf in enumerate(pfull):
        if pf > source_level_pressure:
            klevel_of_source = k
            break
    source_level = np.zeros(lat.shape).astype(int)
    kmax = 39
    for j, la in enumerate(lat):
        source_level[j] = (kmax + 1) - (
            (kmax + 1 - klevel_of_source) * np.cos(lat[j]) + 0.5
        )

    urange_fid = h5py.File(h5str, "a")
    dataloc = f"{year:02d}/"
    try:
        uran = urange_fid[dataloc][()]
        urange_fid.close()
        return uran
    except KeyError:
        U = load_year(year)["ucomp"][()]
        uran = np.zeros((1440, 64, 128), dtype=np.float32)
        for j in range(lat.shape[0]):
            sl = source_level[j]
            uran[:, j, :] = np.max(U[:, :sl, j, :], axis=1) - np.min(
                U[:, :sl, j, :], axis=1
            )
        urange_fid[dataloc] = uran
        urange_fid.close()
        return uran


def save_ao_getstats(h5str, stdim, years):
    if type(years) == int:
        verify = False
        return get_stats_oneyear(h5str, stdim, years, verify)
    elif type(years) == list:
        stats_fid = h5py.File(h5str, "a", swmr=True)
        # stats_fid.swmr_mode = True
        # First, make sure stats are already present for each year. If they don't exist, save them in h5str[dataloc].
        verify = True
        for y, yr in enumerate(years):
            if y == 0:
                stats = get_stats_oneyear(h5str, stdim, yr, False)
                for var in VAR_NAMES:
                    stats[f"{var}_std"] = stats[f"{var}_std"] ** 2
            else:
                get_stats_oneyear(h5str, stdim, yr, verify)
        # Get average of the annual stats.
        for y, yr in enumerate(years):
            dataloc = f"{yr:02d}/stdim{stdim:d}"
            if y > 0:
                for var in VAR_NAMES:
                    stats[f"{var}_mean"] += stats_fid[f"{dataloc}/{var}/mean"][()]
                    stats[f"{var}_std"] += stats_fid[f"{dataloc}/{var}/std"][()] ** 2
        for var in VAR_NAMES:
            stats[f"{var}_mean"] /= len(years)
            stats[f"{var}_std"] = np.sqrt(stats[f"{var}_std"] / len(years))
        return stats


def save_ao_get_tlvars(h5str, year):
    tl_fid = h5py.File(h5str, "a")
    if type(year) == int:
        dataloc = f"{year:02d}"
        try:
            x = tl_fid[dataloc][()]
            tl_fid.close()
            return x
        except KeyError:
            tl_vars = get_tlvars(year)
            tl_fid[dataloc] = tl_vars
            tl_fid.close()
            return tl_vars
    elif type(year) == list:
        # First, make sure stats are already present for each year. If they don't exist, save them in h5str/dataloc.
        for yr in year:
            dataloc = f"{yr:02d}"
            try:
                tl_fid[dataloc]
            except KeyError:
                tl_vars = get_tlvars(yr)
                tl_fid[dataloc] = tl_vars
        # Get average of the annual stats.
        for y, yr in enumerate(year):
            dataloc = f"{yr:02d}"
            if y == 0:
                tl_var = tl_fid[dataloc][()]
            else:
                tl_var += tl_fid[dataloc][()]
        return tl_var / len(year)


def save_ao_get_annualcycle(h5str, years):
    ac_fid = h5py.File(h5str, "a")
    try:
        gU = ac_fid[f"{years[0]:02d}to{years[-1]:02d}/gU"][()]
        gV = ac_fid[f"{years[0]:02d}to{years[-1]:02d}/gV"][()]
        ac_fid.close()
        return [gU, gV]
    except KeyError:
        gU = get_annualcycle(years, "gU")
        gV = get_annualcycle(years, "gV")
        ac_fid[f"{years[0]:02d}to{years[-1]:02d}/gU"] = gU
        ac_fid[f"{years[0]:02d}to{years[-1]:02d}/gV"] = gV
        ac_fid.close()
        return [gU, gV]


def standardize(stats, data):
    for var in VAR_NAMES:
        data[var] -= stats[f"{var}_mean"]  # [:data[var].shape[-1]]
        data[var] /= stats[f"{var}_std"]  # [:data[var].shape[-1]]


def standardize_batch(stats, dim, batch):
    input_3d, SP, target = batch
    if dim == 0:
        input_3d -= stats["input_3d"][0, :]
        input_3d /= stats["input_3d"][1, :]
        SP -= stats["input_loc"][0]
        SP /= stats["input_loc"][1]
        target -= stats["target"][0, :]
        target /= stats["target"][1, :]
        return input_3d, SP, target
    elif dim == 1:
        input_3d -= stats["input_3d"][0, :, :].reshape(1, 40, 1, 1, 4)
        input_3d /= stats["input_3d"][1, :, :].reshape(1, 40, 1, 1, 4)
        SP -= stats["input_loc"][0]
        SP /= stats["input_loc"][1]
        target -= stats["target"][0, :, :].reshape(1, 40, 1, 1, 2)
        target /= stats["target"][1, :, :].reshape(1, 40, 1, 1, 2)
        return input_3d, SP, target


def standardize_acgUV(ac_gU, ac_gV, stats):
    ac_gU -= stats["target"][0, 0]
    ac_gU /= stats["target"][1, 0]
    ac_gV -= stats["target"][0, 1]
    ac_gV /= stats["target"][1, 1]
    return ac_gU, ac_gV


def standardize_std(stats, dim, load2mem, sample):
    input_3d, input_loc, target = sample
    if load2mem is True:
        if dim == 0:
            input_3d /= stats["input_3d"][1, :]
            input_loc[2:] /= stats["input_loc"][1]
            target /= stats["target"][1, :]
        elif dim == 1:
            input_3d /= stats["input_3d"][1, :, :]
            input_loc[2:] /= stats["input_loc"][1]
            target /= stats["target"][1, :, :]
        return input_3d, input_loc, target
    else:
        if dim == 0:
            input_3d /= stats["input_3d"][1, :]
            input_loc[:, 2:] /= 100
            input_loc[:, 2:] /= stats["input_loc"][1]
            target /= stats["target"][1, :]
        elif dim == 1:
            input_3d /= stats["input_3d"][1, :, :]
            input_loc[:, 2:] /= 100
            input_loc[:, 2:] /= stats["input_loc"][1]
            target /= stats["target"][1, :, :]
        return input_3d, input_loc, target


def test_batch(
    data, idx, index_info, batch_alloc, model, loss_fn, device, cfunc, metrics=None
):
    idxst, idxend = idx
    bs = idxend - idxst
    with torch.no_grad():
        # Get batch.
        x3, xloc, gUV = cfunc(data, idxst, idxend, index_info, batch_alloc)
        x3, xloc, gUV = x3.to(device), xloc.to(device), gUV.to(device)

        # Compute prediction error
        pred_gUV = model([x3[:bs], xloc[:bs, :3]])
        loss = loss_fn(pred_gUV, gUV[:bs])
        if metrics:
            # Metrics
            metric_values = {}
            for metric_key in metrics.keys():
                metric_values[metric_key] = metrics[metric_key](gUV[:bs], pred_gUV)
            return loss, rel2norm
        rel2norm = torch.mean(
            torch.linalg.vector_norm(pred_gUV - gUV[:bs], dim=1)
            / torch.linalg.vector_norm(gUV[:bs], dim=1)
        )
        return loss, rel2norm


def test_batch2(
    data, idx, index_info, batch_alloc, model, loss_fn, device, cfunc, metrics=None
):
    idxst, idxend = idx
    bs = idxend - idxst
    with torch.no_grad():
        # Get batch.
        x3, xloc, gUV, pgU = cfunc(data, idxst, idxend, index_info, batch_alloc)
        x3, xloc, gUV, pgU = (
            x3.to(device),
            xloc.to(device),
            gUV.to(device),
            pgU.to(device),
        )

        # Compute prediction error
        pred_gUV = model([x3[:bs], pgU[:bs]])
        loss = loss_fn(pred_gUV, gUV[:bs])
        if metrics:
            # Metrics
            metric_values = {}
            for metric_key in metrics.keys():
                metric_values[metric_key] = metrics[metric_key](gUV[:bs], pred_gUV)
            return loss, rel2norm
        rel2norm = torch.mean(
            torch.linalg.vector_norm(pred_gUV - gUV[:bs], dim=1)
            / torch.linalg.vector_norm(gUV[:bs], dim=1)
        )
        return loss, rel2norm


def train_batch(
    data,idx,index_info,batch_alloc,
    model,loss_fn,optimizer,device,cfunc,
    weightedLoss=False,
    metrics=None,
    l1_amp=0,
):
    tic = perf_counter()
    idxst, idxend = idx
    bs = idxend - idxst
    x3, xloc, gUV = cfunc(data, idxst, idxend, index_info, batch_alloc)
    x3, xloc, gUV = x3.to(device), xloc.to(device), gUV.to(device)

    # Compute prediction error
    pred_gUV = model([x3[:bs], xloc[:bs, :3]])
    if weightedLoss is True:
        # weights are located in the last index of xloc.
        pred_gUV = pred_gUV * xloc[:bs, 3]
        gUV[:bs] = gUV[:bs] * xloc[:bs, 3]
    loss0 = loss_fn(pred_gUV, gUV[:bs])
    if l1_amp == 0:
        l1_reg = 0
    else:
        l1_reg = add_l1_reg(model, l1_amp)
    loss = loss0 + l1_amp * l1_reg

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    toc = perf_counter()
    # Metrics
    if metrics:
        metric_values = {}
        for metric_key in metrics.keys():
            metric_values[metric_key] = metrics[metric_key](gUV[:bs], pred_gUV)
        return loss, metric_values
    else:
        rel2norm = torch.mean(
            torch.linalg.vector_norm(pred_gUV - gUV[:bs], dim=1)
            / torch.linalg.vector_norm(gUV[:bs], dim=1)
        )
        info(
            f"Train loss : {loss0:>7e}, rel2norm:{rel2norm:>7e}, {toc-tic:>5f} (sec)",
            end="",
        )
        return loss, rel2norm


def train_batch2(
    data,
    idx,
    index_info,
    batch_alloc,
    model,
    loss_fn,
    optimizer,
    device,
    cfunc,
    weightedLoss=False,
    metrics=None,
    l1_amp=0,
):
    tic = perf_counter()
    idxst, idxend = idx
    bs = idxend - idxst
    x3, xloc, gUV, pgU = cfunc(data, idxst, idxend, index_info, batch_alloc)
    x3, xloc, gUV, pgU = x3.to(device), xloc.to(device), gUV.to(device), pgU.to(device)

    # Compute prediction error
    pred_gUV = model([x3[:bs], pgU[:bs]])
    if weightedLoss is True:
        # weights are located in the last index of xloc.
        pred_gUV = pred_gUV * xloc[:bs]
        gUV[:bs] = gUV[:bs] * xloc[:bs]
    loss0 = loss_fn(pred_gUV, gUV[:bs])
    if l1_amp == 0:
        l1_reg = 0
    else:
        l1_reg = add_l1_reg(model, l1_amp)
    loss = loss0 + l1_amp * l1_reg

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    toc = perf_counter()
    # Metrics
    if metrics:
        metric_values = {}
        for metric_key in metrics.keys():
            metric_values[metric_key] = metrics[metric_key](gUV[:bs], pred_gUV)
        return loss, metric_values
    rel2norm = torch.mean(
        torch.linalg.vector_norm(pred_gUV - gUV[:bs], dim=1)
        / torch.linalg.vector_norm(gUV[:bs], dim=1)
    )
    info(
        f"Train loss : {loss0:>7e}, rel2norm:{rel2norm_va:>7e}, {toc-tic:>5f} (sec)",
        end="",
    )
    return loss, rel2norm


def train_loop(data, model, loss_fn, optimizer, scheduler, metrics, train_info, funcs):
    """
    data        : list/tuple of [trn, tst]
    model       : an instance of ML architecture
    loss_fn     : loss function
    optimizer   : optimizer
    train_info  : init_epoch    : initial epoch,
                  bs            : batch size,
                  num_steps     : number of batches to train with before validating.
                  val_batches   : number batches to compute validation.
                  epochs        : total number of batches to train.
                  save_loc      : where to save weights & tensorboard data.
                  name          : name of this experiment
                  patience      : early stop if val loss doesn't improve for {patience} batches of training.
                  portion       : size of validation set as a portion of a year's worth of data.
                  device        : cpu or cuda
                  l1_amp        : l1 regularization amplitude
                  save_versions : Bool, False updates weights, True saves weights at every improvement.
                  transform     : how to standardize data
    """
    # Unpack inputs.
    trn_yrs, tst_yr = data
    (
        init_epoch,
        bs,
        num_steps,
        val_batches,
        epochs,
        save_loc,
        name,
        patience,
        portion,
        device,
        l1_amp,
        save_versions,
        transform,
        batch_alloc,
    ) = train_info
    get_batch, get_data = funcs
    # Set up tensorboard writers.
    for fname in ["train", "test"]:
        if os.path.isdir(save_loc + name + fname) is False:
            os.mkdir(save_loc + name + fname)
    writer_tr = SummaryWriter(log_dir=f"{save_loc}{name}train/")
    writer_va = SummaryWriter(log_dir=f"{save_loc}{name}test/")

    n_latlon = 64 * 128
    n_window = 1440
    N_tr = n_latlon * n_window
    N_va = (n_latlon * n_window) // portion + 1
    patience = N_tr // bs * patience
    save_name = save_loc + name + "ae.h5"

    # Counters
    lastbatch_tr = N_tr // bs + 1
    lastbatch_va = N_va // bs + 1
    val_batches = lastbatch_va if val_batches == 0 else val_batches
    epoch = init_epoch
    pass_tr = init_epoch % lastbatch_tr
    pass_va = (init_epoch // num_steps) * val_batches % lastbatch_va
    baseline_loss = 0
    trn = {}
    lbvl = init_epoch  # epoch of last best validation loss
    y_tr = 0
    info(f"Loading test data: year {tst_yr:02d}.")
    tst = get_data(tst_yr, transform=transform)
    N_sample = 0
    save_every = n_latlon * (1440 // 12)  # monthly.
    while epoch <= epochs:
        # Shuffle when entire training/validation set have been passed through, and load in new trn years if necessary.
        if pass_tr == 0 or epoch == init_epoch:
            idxinds_tr = (
                np.random.permutation(N_tr)
                if epoch == init_epoch
                else np.random.permutation(idxinds_tr)
            )
            index_info_tr = get_la_lo_tstart(idxinds_tr)
            info(f"Loading train data: year {trn_yrs[y_tr]:02d}.")
            trn = get_data(trn_yrs[y_tr], transform=transform, data=trn)
            y_tr = y_tr + 1 if y_tr < len(trn_yrs) - 1 else 0
        if pass_va == 0 or epoch == init_epoch:
            idxinds_va = (
                np.random.choice(np.arange(0, N_tr), N_va, replace=False)
                if epoch == init_epoch
                else np.random.permutation(idxinds_va)
            )
            index_info_va = get_la_lo_tstart(idxinds_va)

        idxst = pass_tr * bs
        idxend = (pass_tr + 1) * bs if pass_tr != lastbatch_tr - 1 else N_tr
        N_sample += idxend - idxst
        info(f"Batch # {epoch}, month#{N_sample//save_every}: ", end="")
        loss_tr, rel2norm_tr = train_batch(
            trn,
            [idxst, idxend],
            index_info_tr,
            batch_alloc,
            model,
            loss_fn,
            optimizer,
            device,
            get_batch,
            l1_amp=l1_amp,
        )
        writer_tr.add_scalar("Loss", loss_tr, N_sample // save_every)
        writer_tr.flush()
        writer_tr.add_scalar("Rel2Norm", rel2norm_tr, N_sample // save_every)
        writer_tr.flush()
        pass_tr += 1
        if epoch % num_steps == 0:
            cum_loss = 0
            cum_rel2norm = 0
            tic = perf_counter()
            for j in range(val_batches):
                idxst = pass_va * bs
                idxend = (pass_va + 1) * bs if pass_va != lastbatch_va - 1 else N_va
                l, r = test_batch(
                    tst,
                    [idxst, idxend],
                    index_info_va,
                    batch_alloc,
                    model,
                    loss_fn,
                    device,
                    get_batch,
                )
                cum_loss += l
                cum_rel2norm += r
                pass_va += 1
                if pass_va == lastbatch_va:
                    if val_batches != lastbatch_va:
                        info("Completed a pass through validation set!")
                    pass_va = 0
            validation_loss = cum_loss / val_batches
            rel2norm_va = cum_rel2norm / val_batches
            toc = perf_counter()
            info(
                f", Validation loss : {validation_loss:>7e}, rel2norm:{rel2norm_va:>7e}, {toc-tic:>5f} (sec)."
            )
            writer_va.add_scalar("Loss", validation_loss, N_sample // save_every)
            writer_va.flush()
            writer_va.add_scalar("Rel2Norm", rel2norm_va, N_sample // save_every)
            writer_va.flush()
            if epoch == init_epoch or baseline_loss == 0:
                baseline_loss = rel2norm_va
                info("Saving initial version of this training segment.")
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    h5str,
                )
            elif baseline_loss > rel2norm_va:
                lbvl = epoch
                info(
                    f"Validation relative norm has improved from {baseline_loss:1.4e} to {rel2norm_va:1.4e} and therefore saving the model."
                )
                baseline_loss = rel2norm_va  # validation_loss
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    h5str,
                )
        else:
            info(".")
        if pass_tr == lastbatch_tr:
            info("Completed a pass through training set!")
            pass_tr = 0
        if epoch - lbvl > patience:
            info(f"Early stop: Model has not improved since {lbvl}.")
            return epoch
        epoch += 1
    return epoch


def train_loop2(
    data,
    model,
    loss_fn,
    optimizer,
    scheduler,
    metrics,
    train_info,
    funcs,
    sampling=["original", 0],
):
    """
    data        : list/tuple of [trn, tst]
    model       : an instance of ML architecture
    loss_fn     : loss function
    optimizer   : optimizer
    train_info  : init_epoch    : initial epoch,
                  bs            : batch size,
                  num_steps     : number of batches to train with before validating.
                  val_batches   : number batches to compute validation.
                  epochs        : total number of batches to train.
                  save_loc      : where to save weights & tensorboard data.
                  name          : name of this experiment
                  patience      : early stop if val loss doesn't improve for {patience} batches of training.
                  portion       : size of validation set as a portion of a year's worth of data.
                  device        : cpu or cuda
                  l1_amp        : l1 regularization amplitude
                  save_versions : Bool, False updates weights, True saves weights at every improvement.
                  transform     : how to standardize data
    """
    # Unpack inputs.
    trn_yrs, tst_yr = data
    (
        init_epoch,
        bs,
        num_steps,
        val_batches,
        epochs,
        save_loc,
        name,
        patience,
        portion,
        device,
        l1_amp,
        save_versions,
        transform,
        batch_alloc,
    ) = train_info
    get_batch, get_data = funcs
    sample_kind, s_funcparams = sampling

    # Set up tensorboard writers.
    for fname in ["train", "test"]:
        if os.path.isdir(save_loc + name + fname) is False:
            os.mkdir(save_loc + name + fname)
    writer_tr = SummaryWriter(log_dir=f"{save_loc}{name}train/")
    writer_va = SummaryWriter(log_dir=f"{save_loc}{name}test/")

    n_latlon = 64 * 128
    n_window = 1440
    N_tr = n_latlon * n_window
    N_va = (n_latlon * n_window) // portion + 1
    patience = N_tr // bs * patience
    save_name = save_loc + name + "ae.h5"

    # Counters
    lastbatch_tr = N_tr // bs + 1
    lastbatch_va = N_va // bs + 1
    val_batches = lastbatch_va if val_batches == 0 else val_batches
    epoch = init_epoch[0] if isinstance(init_epoch, list) else init_epoch
    pass_tr = epoch % lastbatch_tr
    pass_va = (epoch // num_steps) * val_batches % lastbatch_va
    baseline_loss = 0
    trn = {}
    lbvl = epoch  # epoch of last best validation loss
    y_tr = 0
    info(f"Loading test data: year {tst_yr:02d}.")
    tst = get_data(tst_yr, transform=transform)
    save_every = n_latlon * (1440 // 12)  # monthly.
    N_sample = init_epoch[1] * save_every if isinstance(init_epoch, list) else 0
    init_epoch = epoch
    while epoch <= epochs:
        # Shuffle when entire training/validation set have been passed through, and load in new trn years if necessary.
        if pass_tr == 0 or epoch == init_epoch:
            if sample_kind == "original":
                idxinds_tr = (
                    np.random.permutation(N_tr)
                    if epoch == init_epoch
                    else np.random.permutation(idxinds_tr)
                )
            elif sample_kind == "histeq":
                functional, nbins, t, maxrepeat = s_funcparams
                idxinds_tr = DB_get_bins_ind_alpha(
                    functional, trn_yrs[y_tr], nbins, t, maxrepeat
                )
                N_tr = idxinds_tr.shape[0]
                lastbatch_tr = N_tr // bs + 1
                pass_tr = epoch % lastbatch_tr if pass_tr != 0 else 0
            index_info_tr = get_la_lo_tstart(idxinds_tr)
            info(f"Loading train data: year {trn_yrs[y_tr]:02d}.")
            trn = get_data(trn_yrs[y_tr], transform=transform, data=trn)
            y_tr = y_tr + 1 if y_tr < len(trn_yrs) - 1 else 0
        if pass_va == 0 or epoch == init_epoch:
            idxinds_va = (
                np.random.choice(np.arange(0, N_tr), N_va, replace=False)
                if epoch == init_epoch
                else np.random.permutation(idxinds_va)
            )
            index_info_va = get_la_lo_tstart(idxinds_va)

        idxst = pass_tr * bs
        idxend = (pass_tr + 1) * bs if pass_tr != lastbatch_tr - 1 else N_tr
        N_sample += idxend - idxst
        info(f"Batch # {epoch}, month#{N_sample//save_every}: ", end="")
        loss_tr, rel2norm_tr = train_batch(
            trn,
            [idxst, idxend],
            index_info_tr,
            batch_alloc,
            model,
            loss_fn,
            optimizer,
            device,
            get_batch,
            l1_amp=l1_amp,
        )
        writer_tr.add_scalar("Loss", loss_tr, N_sample // save_every)
        writer_tr.flush()
        writer_tr.add_scalar("Rel2Norm", rel2norm_tr, N_sample // save_every)
        writer_tr.flush()
        pass_tr += 1
        if epoch % num_steps == 0:
            cum_loss = 0
            cum_rel2norm = 0
            tic = perf_counter()
            for j in range(val_batches):
                idxst = pass_va * bs
                idxend = (pass_va + 1) * bs if pass_va != lastbatch_va - 1 else N_va
                l, r = test_batch(
                    tst,
                    [idxst, idxend],
                    index_info_va,
                    batch_alloc,
                    model,
                    loss_fn,
                    device,
                    get_batch,
                )
                cum_loss += l
                cum_rel2norm += r
                pass_va += 1
                if pass_va == lastbatch_va:
                    if val_batches != lastbatch_va:
                        info("Completed a pass through validation set!")
                    pass_va = 0
            validation_loss = cum_loss / val_batches
            rel2norm_va = cum_rel2norm / val_batches
            toc = perf_counter()
            info(
                f", Validation loss : {validation_loss:>7e}, rel2norm:{rel2norm_va:>7e}, {toc-tic:>5f} (sec)."
            )
            writer_va.add_scalar("Loss", validation_loss, N_sample // save_every)
            writer_va.flush()
            writer_va.add_scalar("Rel2Norm", rel2norm_va, N_sample // save_every)
            writer_va.flush()
            if epoch == init_epoch or baseline_loss == 0:
                baseline_loss = rel2norm_va
                info("Saving initial version of this training segment.")
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    h5str,
                )
            elif baseline_loss > rel2norm_va:
                lbvl = epoch
                info(
                    f"Validation relative norm has improved from {baseline_loss:1.4e} to {rel2norm_va:1.4e} and therefore saving the model."
                )
                baseline_loss = rel2norm_va  # validation_loss
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    h5str,
                )
        else:
            info(".")
        if pass_tr == lastbatch_tr:
            info("Completed a pass through training set!")
            pass_tr = 0
        if epoch - lbvl > patience:
            info(f"Early stop: Model has not improved since {lbvl}.")
            return epoch
        epoch += 1
    return epoch


def train_loop3(
    data,
    model,
    loss_fn,
    optimizer,
    scheduler,
    metrics,
    train_info,
    funcs,
    sampling=["original", 0],
    lbvl=0,
):
    """
    data        : list/tuple of [trn, tst]
    model       : an instance of ML architecture
    loss_fn     : loss function
    optimizer   : optimizer
    train_info  : init_epoch    : initial epoch,
                  bs            : batch size,
                  num_steps     : number of batches to train with before validating.
                  val_batches   : number batches to compute validation.
                  epochs        : total number of batches to train.
                  save_loc      : where to save weights & tensorboard data.
                  name          : name of this experiment
                  patience      : early stop if val loss doesn't improve for {patience} batches of training.
                  portion       : size of validation set as a portion of a year's worth of data.
                  device        : cpu or cuda
                  l1_amp        : l1 regularization amplitude
                  save_versions : Bool, False updates weights, True saves weights at every improvement.
                  transform     : how to standardize data
    """
    # Unpack inputs.
    trn_yrs, tst_yr = data
    (
        init_epoch,
        bs,
        num_steps,
        val_batches,
        epochs,
        save_loc,
        name,
        patience,
        portion,
        device,
        l1_amp,
        save_versions,
        transform,
        batch_alloc,
    ) = train_info
    get_batch, get_data = funcs
    sample_kind, s_funcparams = sampling

    # Set up tensorboard writers.
    for xstep in ["sample/", "step/"]:
        if os.path.isdir(f"{save_loc}{name}{xstep}") is False:
            os.mkdir(f"{save_loc}{name}{xstep}")
        for fname in ["train", "test"]:
            if os.path.isdir(f"{save_loc}{name}{xstep}{fname}/") is False:
                os.mkdir(f"{save_loc}{name}{xstep}{fname}/")
    writer_tr_sa = SummaryWriter(log_dir=f"{save_loc}{name}sample/train/")
    writer_va_sa = SummaryWriter(log_dir=f"{save_loc}{name}sample/test/")
    writer_tr_st = SummaryWriter(log_dir=f"{save_loc}{name}step/train/")
    writer_va_st = SummaryWriter(log_dir=f"{save_loc}{name}step/test/")

    n_latlon = 64 * 128
    n_window = 1440
    N_tr = n_latlon * n_window
    N_va = (n_latlon * n_window) // portion + 1
    patience = N_tr // bs * patience
    save_name = save_loc + name + "ae.h5"

    # Counters
    lastbatch_tr = N_tr // bs + 1
    lastbatch_va = N_va // bs + 1
    val_batches = lastbatch_va if val_batches == 0 else val_batches
    epoch = init_epoch[0] if isinstance(init_epoch, list) else init_epoch
    pass_tr = epoch % lastbatch_tr
    pass_va = (epoch // num_steps) * val_batches % lastbatch_va
    baseline_loss = 0
    trn = {}
    lbvl = epoch if lbvl == 0 else lbvl  # epoch of last best validation loss
    y_tr = 0
    info(f"Loading test data: year {tst_yr:02d}.")
    tst = get_data(tst_yr, transform=transform)
    save_every = n_latlon * (1440 // 12)  # monthly.
    N_sample = init_epoch[1] * save_every if isinstance(init_epoch, list) else 0
    init_epoch = epoch
    while epoch <= epochs:
        # Shuffle when entire training/validation set have been passed through, and load in new trn years if necessary.
        if pass_tr == 0 or epoch == init_epoch:
            if sample_kind == "original":
                info("No sampling strategy!")
                idxinds_tr = (
                    np.random.permutation(N_tr)
                    if epoch == init_epoch
                    else np.random.permutation(idxinds_tr)
                )
            elif sample_kind == "histeq":
                functional, nbins, t, maxrepeat = s_funcparams
                info(
                    f"Yes sampling strategy:nbins={nbins}, t={t}, maxrepeat={maxrepeat}"
                )
                idxinds_tr = DB_get_bins_ind_alpha(
                    functional, trn_yrs[y_tr], nbins, t, maxrepeat
                )
                N_tr = idxinds_tr.shape[0]
                lastbatch_tr = N_tr // bs + 1
                pass_tr = epoch % lastbatch_tr if pass_tr != 0 else 0
            index_info_tr = get_la_lo_tstart(idxinds_tr)
            info(f"Loading train data: year {trn_yrs[y_tr]:02d}.")
            trn = get_data(trn_yrs[y_tr], transform=transform, data=trn)
            y_tr = y_tr + 1 if y_tr < len(trn_yrs) - 1 else 0
        if pass_va == 0 or epoch == init_epoch:
            idxinds_va = (
                np.random.choice(np.arange(0, N_tr), N_va, replace=False)
                if epoch == init_epoch
                else np.random.permutation(idxinds_va)
            )
            index_info_va = get_la_lo_tstart(idxinds_va)

        idxst = pass_tr * bs
        idxend = (pass_tr + 1) * bs if pass_tr != lastbatch_tr - 1 else N_tr
        N_sample += idxend - idxst
        info(f"Batch # {epoch}, month#{N_sample//save_every}: ", end="")
        loss_tr, rel2norm_tr = train_batch(
            trn,
            [idxst, idxend],
            index_info_tr,
            batch_alloc,
            model,
            loss_fn,
            optimizer,
            device,
            get_batch,
            l1_amp=l1_amp,
        )
        writer_tr_sa.add_scalar("Loss", loss_tr, N_sample // save_every)
        writer_tr_sa.flush()
        writer_tr_sa.add_scalar("Rel2Norm", rel2norm_tr, N_sample // save_every)
        writer_tr_sa.flush()
        writer_tr_st.add_scalar("Loss", loss_tr, epoch)
        writer_tr_st.flush()
        writer_tr_st.add_scalar("Rel2Norm", rel2norm_tr, epoch)
        writer_tr_st.flush()
        pass_tr += 1
        if epoch % num_steps == 0 or epoch == init_epoch:
            cum_loss = 0
            cum_rel2norm = 0
            tic = perf_counter()
            for j in range(val_batches):
                idxst = pass_va * bs
                idxend = (pass_va + 1) * bs if pass_va != lastbatch_va - 1 else N_va
                l, r = test_batch(
                    tst,
                    [idxst, idxend],
                    index_info_va,
                    batch_alloc,
                    model,
                    loss_fn,
                    device,
                    get_batch,
                )
                cum_loss += l
                cum_rel2norm += r
                pass_va += 1
                if pass_va == lastbatch_va:
                    if val_batches != lastbatch_va:
                        info("Completed a pass through validation set!")
                    pass_va = 0
            validation_loss = cum_loss / val_batches
            rel2norm_va = cum_rel2norm / val_batches
            toc = perf_counter()
            info(
                f", Validation loss : {validation_loss:>7e}, rel2norm:{rel2norm_va:>7e}, {toc-tic:>5f} (sec)."
            )
            writer_va_sa.add_scalar("Loss", validation_loss, N_sample // save_every)
            writer_va_sa.flush()
            writer_va_sa.add_scalar("Rel2Norm", rel2norm_va, N_sample // save_every)
            writer_va_sa.flush()
            writer_va_st.add_scalar("Loss", validation_loss, epoch)
            writer_va_st.flush()
            writer_va_st.add_scalar("Rel2Norm", rel2norm_va, epoch)
            writer_va_st.flush()
            if epoch == init_epoch or baseline_loss == 0:
                baseline_loss = rel2norm_va
                info("Saving initial version of this training segment.")
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    h5str,
                )
            elif baseline_loss > rel2norm_va:
                lbvl = epoch
                info(
                    f"Validation relative norm has improved from {baseline_loss:1.4e} to {rel2norm_va:1.4e} and therefore saving the model."
                )
                baseline_loss = rel2norm_va  # validation_loss
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    h5str,
                )
        else:
            info(".")
        if pass_tr == lastbatch_tr:
            info("Completed a pass through training set!")
            pass_tr = 0
        if epoch - lbvl > patience:
            info(f"Early stop: Model has not improved since {lbvl}.")
            return epoch
        epoch += 1
    return epoch


def train_loop4(
    data,
    model,
    loss_fn,
    optimizer,
    scheduler,
    metrics,
    train_info,
    funcs,
    sampling=["original", 0],
    lbvl=0,
):
    """
    weightedloss
    data        : list/tuple of [trn, tst]
    model       : an instance of ML architecture
    loss_fn     : loss function
    optimizer   : optimizer
    train_info  : init_epoch    : initial epoch,
                  bs            : batch size,
                  num_steps     : number of batches to train with before validating.
                  val_batches   : number batches to compute validation.
                  epochs        : total number of batches to train.
                  save_loc      : where to save weights & tensorboard data.
                  name          : name of this experiment
                  patience      : early stop if val loss doesn't improve for {patience} batches of training.
                  portion       : size of validation set as a portion of a year's worth of data.
                  device        : cpu or cuda
                  l1_amp        : l1 regularization amplitude
                  save_versions : Bool, False updates weights, True saves weights at every improvement.
                  transform     : how to standardize data
    """
    # Unpack inputs.
    trn_yrs, tst_yr = data
    (
        init_epoch,
        bs,
        num_steps,
        val_batches,
        epochs,
        save_loc,
        name,
        patience,
        portion,
        device,
        l1_amp,
        save_versions,
        transform,
        batch_alloc,
        weightedLoss,
    ) = train_info
    weightedLoss = False if "weightedLoss" not in globals() else weightedLoss
    get_batch, get_data = funcs
    sample_kind, s_funcparams = sampling

    # Set up tensorboard writers.
    for xstep in ["sample/", "step/"]:
        if os.path.isdir(f"{save_loc}{name}{xstep}") is False:
            os.mkdir(f"{save_loc}{name}{xstep}")
        for fname in ["train", "test"]:
            if os.path.isdir(f"{save_loc}{name}{xstep}{fname}/") is False:
                os.mkdir(f"{save_loc}{name}{xstep}{fname}/")
    writer_tr_sa = SummaryWriter(log_dir=f"{save_loc}{name}sample/train/")
    writer_va_sa = SummaryWriter(log_dir=f"{save_loc}{name}sample/test/")
    writer_tr_st = SummaryWriter(log_dir=f"{save_loc}{name}step/train/")
    writer_va_st = SummaryWriter(log_dir=f"{save_loc}{name}step/test/")

    n_latlon = 64 * 128
    n_window = 1440
    N_tr = n_latlon * n_window
    N_va = (n_latlon * n_window) // portion + 1
    patience = N_tr // bs * patience
    save_name = save_loc + name + "ae.h5"

    # Counters
    lastbatch_tr = N_tr // bs + 1
    lastbatch_va = N_va // bs + 1
    val_batches = lastbatch_va if val_batches == 0 else val_batches
    epoch = init_epoch[0] if isinstance(init_epoch, list) else init_epoch
    pass_tr = epoch % lastbatch_tr
    pass_va = (epoch // num_steps) * val_batches % lastbatch_va
    baseline_loss = 0
    trn = {}
    lbvl = epoch if lbvl == 0 else lbvl  # epoch of last best validation loss
    y_tr = 0
    info(f"Loading test data: year {tst_yr:02d}.")
    tst = get_data(tst_yr, transform=transform)
    save_every = n_latlon * (1440 // 12)  # monthly.
    N_sample = init_epoch[1] * save_every if isinstance(init_epoch, list) else 0
    init_epoch = epoch
    while epoch <= epochs:
        # Shuffle when entire training/validation set have been passed through, and load in new trn years if necessary.
        if pass_tr == 0 or epoch == init_epoch:
            if sample_kind == "original":
                info("No sampling strategy!")
                idxinds_tr = (
                    np.random.permutation(N_tr)
                    if epoch == init_epoch
                    else np.random.permutation(idxinds_tr)
                )
                alphas = 1
            elif sample_kind == "histeq":
                functional, nbins, t, maxrepeat = s_funcparams
                info(
                    f"Yes sampling strategy:nbins={nbins}, t={t}, maxrepeat={maxrepeat}"
                )
                alphas = DB_get_bins_ind_alpha(
                    functional, trn_yrs[y_tr], nbins, t, maxrepeat, returnalpha=True
                )
                info(alphas.shape)  ###########
                idxinds_tr = (
                    np.random.permutation(N_tr)
                    if epoch == init_epoch
                    else np.random.permutation(idxinds_tr)
                )
                N_tr = idxinds_tr.shape[0]
                lastbatch_tr = N_tr // bs + 1
                pass_tr = epoch % lastbatch_tr if pass_tr != 0 else 0
            index_info_tr = get_la_lo_tstart(idxinds_tr)
            info(f"Loading train data: year {trn_yrs[y_tr]:02d}.")
            trn = get_data(trn_yrs[y_tr], transform=transform, data=trn, alphas=alphas)
            y_tr = y_tr + 1 if y_tr < len(trn_yrs) - 1 else 0
        if pass_va == 0 or epoch == init_epoch:
            idxinds_va = (
                np.random.choice(np.arange(0, N_tr), N_va, replace=False)
                if epoch == init_epoch
                else np.random.permutation(idxinds_va)
            )
            index_info_va = get_la_lo_tstart(idxinds_va)

        idxst = pass_tr * bs
        idxend = (pass_tr + 1) * bs if pass_tr != lastbatch_tr - 1 else N_tr
        N_sample += idxend - idxst
        info(f"Batch # {epoch}, month#{N_sample//save_every}: ", end="")
        loss_tr, rel2norm_tr = train_batch(
            trn,
            [idxst, idxend],
            index_info_tr,
            batch_alloc,
            model,
            loss_fn,
            optimizer,
            device,
            get_batch,
            weightedLoss=weightedLoss,
            l1_amp=l1_amp,
        )
        writer_tr_sa.add_scalar("Loss", loss_tr, N_sample // save_every)
        writer_tr_sa.flush()
        writer_tr_sa.add_scalar("Rel2Norm", rel2norm_tr, N_sample // save_every)
        writer_tr_sa.flush()
        writer_tr_st.add_scalar("Loss", loss_tr, epoch)
        writer_tr_st.flush()
        writer_tr_st.add_scalar("Rel2Norm", rel2norm_tr, epoch)
        writer_tr_st.flush()
        pass_tr += 1
        if epoch % num_steps == 0 or epoch == init_epoch:
            cum_loss = 0
            cum_rel2norm = 0
            tic = perf_counter()
            for _ in range(val_batches):
                idxst = pass_va * bs
                idxend = (pass_va + 1) * bs if pass_va != lastbatch_va - 1 else N_va
                l, r = test_batch(
                    tst,
                    [idxst, idxend],
                    index_info_va,
                    batch_alloc,
                    model,
                    loss_fn,
                    device,
                    get_batch,
                )
                cum_loss += l
                cum_rel2norm += r
                pass_va += 1
                if pass_va == lastbatch_va:
                    if val_batches != lastbatch_va:
                        info("Completed a pass through validation set!")
                    pass_va = 0
            validation_loss = cum_loss / val_batches
            rel2norm_va = cum_rel2norm / val_batches
            toc = perf_counter()
            info(
                f", Validation loss : {validation_loss:>7e}, rel2norm:{rel2norm_va:>7e}, {toc-tic:>5f} (sec)."
            )
            writer_va_sa.add_scalar("Loss", validation_loss, N_sample // save_every)
            writer_va_sa.flush()
            writer_va_sa.add_scalar("Rel2Norm", rel2norm_va, N_sample // save_every)
            writer_va_sa.flush()
            writer_va_st.add_scalar("Loss", validation_loss, epoch)
            writer_va_st.flush()
            writer_va_st.add_scalar("Rel2Norm", rel2norm_va, epoch)
            writer_va_st.flush()
            if epoch == init_epoch or baseline_loss == 0:
                baseline_loss = rel2norm_va
                info("Saving initial version of this training segment.")
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    h5str,
                )
            elif baseline_loss > rel2norm_va:
                lbvl = epoch
                info(
                    f"Validation relative norm has improved from {baseline_loss:1.4e} to {rel2norm_va:1.4e} and therefore saving the model."
                )
                baseline_loss = rel2norm_va  # validation_loss
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    h5str,
                )
        else:
            info(".")
        if pass_tr == lastbatch_tr:
            info("Completed a pass through training set!")
            pass_tr = 0
        if epoch - lbvl > patience:
            info(f"Early stop: Model has not improved since {lbvl}.")
            return epoch
        epoch += 1
    return epoch


def train_loop5(
    data,
    model,
    loss_fn,
    optimizer,
    scheduler,
    metrics,
    train_info,
    funcs,
    sampling=["original", 0],
    lbvl=0,
):
    """
    weighted loss & train/testbatch2
    data        : list/tuple of [trn, tst]
    model       : an instance of ML architecture
    loss_fn     : loss function
    optimizer   : optimizer
    train_info  : init_epoch    : initial epoch,
                  bs            : batch size,
                  num_steps     : number of batches to train with before validating.
                  val_batches   : number batches to compute validation.
                  epochs        : total number of batches to train.
                  save_loc      : where to save weights & tensorboard data.
                  name          : name of this experiment
                  patience      : early stop if val loss doesn't improve for {patience} batches of training.
                  portion       : size of validation set as a portion of a year's worth of data.
                  device        : cpu or cuda
                  l1_amp        : l1 regularization amplitude
                  save_versions : Bool, False updates weights, True saves weights at every improvement.
                  transform     : how to standardize data
    """
    # Unpack inputs.
    trn_yrs, tst_yr = data
    (
        init_epoch,
        bs,
        num_steps,
        val_batches,
        epochs,
        save_loc,
        name,
        patience,
        portion,
        device,
        l1_amp,
        save_versions,
        transform,
        batch_alloc,
        weightedLoss,
    ) = train_info
    weightedLoss = False if "weightedLoss" not in globals() else weightedLoss
    get_batch, get_data = funcs
    sample_kind, s_funcparams = sampling

    # Set up tensorboard writers.
    for xstep in ["sample/", "step/"]:
        if os.path.isdir(f"{save_loc}{name}{xstep}") is False:
            os.mkdir(f"{save_loc}{name}{xstep}")
        for fname in ["train", "test"]:
            if os.path.isdir(f"{save_loc}{name}{xstep}{fname}/") is False:
                os.mkdir(f"{save_loc}{name}{xstep}{fname}/")
    writer_tr_sa = SummaryWriter(log_dir=f"{save_loc}{name}sample/train/")
    writer_va_sa = SummaryWriter(log_dir=f"{save_loc}{name}sample/test/")
    writer_tr_st = SummaryWriter(log_dir=f"{save_loc}{name}step/train/")
    writer_va_st = SummaryWriter(log_dir=f"{save_loc}{name}step/test/")

    n_latlon = 64 * 128
    n_window = 1440
    N_tr = n_latlon * n_window
    N_va = (n_latlon * n_window) // portion + 1
    patience = N_tr // bs * patience
    save_name = save_loc + name + "dnn.h5"

    # Counters
    lastbatch_tr = N_tr // bs + 1
    lastbatch_va = N_va // bs + 1
    val_batches = lastbatch_va if val_batches == 0 else val_batches
    epoch = init_epoch[0] if isinstance(init_epoch, list) else init_epoch
    pass_tr = epoch % lastbatch_tr
    pass_va = (epoch // num_steps) * val_batches % lastbatch_va
    baseline_loss = 0
    trn = {}
    lbvl = epoch if lbvl == 0 else lbvl  # epoch of last best validation loss
    y_tr = 0
    info(f"Loading test data: year {tst_yr:02d}.")
    tst = get_data(tst_yr, transform=transform)
    save_every = n_latlon * (1440 // 12)  # monthly.
    N_sample = init_epoch[1] * save_every if isinstance(init_epoch, list) else 0
    init_epoch = epoch
    while epoch <= epochs:
        # Shuffle when entire training/validation set have been passed through, and load in new trn years if necessary.
        if pass_tr == 0 or epoch == init_epoch:
            if sample_kind == "original":
                info("No sampling strategy!")
                idxinds_tr = (
                    np.random.permutation(N_tr)
                    if epoch == init_epoch
                    else np.random.permutation(idxinds_tr)
                )
                alphas = 1
            elif sample_kind == "histeq":
                functional, nbins, t, maxrepeat = s_funcparams
                info(
                    f"Yes sampling strategy:nbins={nbins}, t={t}, maxrepeat={maxrepeat}"
                )
                alphas = DB_get_bins_ind_alpha(
                    functional, trn_yrs[y_tr], nbins, t, maxrepeat, returnalpha=True
                )
                info(alphas.shape)  ###########
                idxinds_tr = (
                    np.random.permutation(N_tr)
                    if epoch == init_epoch
                    else np.random.permutation(idxinds_tr)
                )
                N_tr = idxinds_tr.shape[0]
                lastbatch_tr = N_tr // bs + 1
                pass_tr = epoch % lastbatch_tr if pass_tr != 0 else 0
            index_info_tr = get_la_lo_tstart(idxinds_tr)
            info(f"Loading train data: year {trn_yrs[y_tr]:02d}.")
            trn = get_data(trn_yrs[y_tr], transform=transform, data=trn, alphas=alphas)
            y_tr = y_tr + 1 if y_tr < len(trn_yrs) - 1 else 0
        if pass_va == 0 or epoch == init_epoch:
            idxinds_va = (
                np.random.choice(np.arange(0, N_tr), N_va, replace=False)
                if epoch == init_epoch
                else np.random.permutation(idxinds_va)
            )
            index_info_va = get_la_lo_tstart(idxinds_va)

        idxst = pass_tr * bs
        idxend = (pass_tr + 1) * bs if pass_tr != lastbatch_tr - 1 else N_tr
        N_sample += idxend - idxst
        info(f"Batch # {epoch}, month#{N_sample//save_every}: ", end="")
        loss_tr, rel2norm_tr = train_batch2(
            trn,
            [idxst, idxend],
            index_info_tr,
            batch_alloc,
            model,
            loss_fn,
            optimizer,
            device,
            get_batch,
            weightedLoss=weightedLoss,
            l1_amp=l1_amp,
        )
        writer_tr_sa.add_scalar("Loss", loss_tr, N_sample // save_every)
        writer_tr_sa.flush()
        writer_tr_sa.add_scalar("Rel2Norm", rel2norm_tr, N_sample // save_every)
        writer_tr_sa.flush()
        writer_tr_st.add_scalar("Loss", loss_tr, epoch)
        writer_tr_st.flush()
        writer_tr_st.add_scalar("Rel2Norm", rel2norm_tr, epoch)
        writer_tr_st.flush()
        pass_tr += 1
        if epoch % num_steps == 0 or epoch == init_epoch:
            cum_loss = 0
            cum_rel2norm = 0
            tic = perf_counter()
            for _ in range(val_batches):
                idxst = pass_va * bs
                idxend = (pass_va + 1) * bs if pass_va != lastbatch_va - 1 else N_va
                l, r = test_batch2(
                    tst,
                    [idxst, idxend],
                    index_info_va,
                    batch_alloc,
                    model,
                    loss_fn,
                    device,
                    get_batch,
                )
                cum_loss += l
                cum_rel2norm += r
                pass_va += 1
                if pass_va == lastbatch_va:
                    if val_batches != lastbatch_va:
                        info("Completed a pass through validation set!")
                    pass_va = 0
            validation_loss = cum_loss / val_batches
            rel2norm_va = cum_rel2norm / val_batches
            toc = perf_counter()
            info(
                f", Validation loss : {validation_loss:>7e}, rel2norm:{rel2norm_va:>7e}, {toc-tic:>5f} (sec)."
            )
            writer_va_sa.add_scalar("Loss", validation_loss, N_sample // save_every)
            writer_va_sa.flush()
            writer_va_sa.add_scalar("Rel2Norm", rel2norm_va, N_sample // save_every)
            writer_va_sa.flush()
            writer_va_st.add_scalar("Loss", validation_loss, epoch)
            writer_va_st.flush()
            writer_va_st.add_scalar("Rel2Norm", rel2norm_va, epoch)
            writer_va_st.flush()
            if epoch == init_epoch or baseline_loss == 0:
                baseline_loss = rel2norm_va
                info("Saving initial version of this training segment.")
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    h5str,
                )
            elif baseline_loss > rel2norm_va:
                lbvl = epoch
                info(
                    f"Validation relative norm has improved from {baseline_loss:1.4e} to {rel2norm_va:1.4e} and therefore saving the model."
                )
                baseline_loss = rel2norm_va  # validation_loss
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    h5str,
                )
        else:
            info(".")
        if pass_tr == lastbatch_tr:
            info("Completed a pass through training set!")
            pass_tr = 0
        if epoch - lbvl > patience:
            info(f"Early stop: Model has not improved since {lbvl}.")
            return epoch
        epoch += 1
    return epoch


def train_loop6(
    data, model,
    loss_fn, optimizer,
    scheduler,metrics,
    train_info,
    funcs,
    sampling=["original", 0],
    lbvl=0,
):
    """
    test/trainbatch 2 (NO weightedloss)
    data        : list/tuple of [trn, tst]
    model       : an instance of ML architecture
    loss_fn     : loss function
    optimizer   : optimizer
    train_info  : init_epoch    : initial epoch,
                  bs            : batch size,
                  num_steps     : number of batches to train with before validating.
                  val_batches   : number batches to compute validation.
                  epochs        : total number of batches to train.
                  save_loc      : where to save weights & tensorboard data.
                  name          : name of this experiment
                  patience      : early stop if val loss doesn't improve for {patience} batches of training.
                  portion       : size of validation set as a portion of a year's worth of data.
                  device        : cpu or cuda
                  l1_amp        : l1 regularization amplitude
                  save_versions : Bool, False updates weights, True saves weights at every improvement.
                  transform     : how to standardize data
    """
    # Unpack inputs.
    trn_yrs, tst_yr = data
    (
        init_epoch,
        bs,
        num_steps,
        val_batches,
        epochs,
        save_loc,
        name,
        patience,
        portion,
        device,
        l1_amp,
        save_versions,
        transform,
        batch_alloc,
    ) = train_info
    get_batch, get_data = funcs
    sample_kind, s_funcparams = sampling

    # Set up tensorboard writers.
    for xstep in ["sample/", "step/"]:
        if os.path.isdir(f"{save_loc}{name}{xstep}") is False:
            os.mkdir(f"{save_loc}{name}{xstep}")
        for fname in ["train", "test"]:
            if os.path.isdir(f"{save_loc}{name}{xstep}{fname}/") is False:
                os.mkdir(f"{save_loc}{name}{xstep}{fname}/")
    writer_tr_sa = SummaryWriter(log_dir=f"{save_loc}{name}sample/train/")
    writer_va_sa = SummaryWriter(log_dir=f"{save_loc}{name}sample/test/")
    writer_tr_st = SummaryWriter(log_dir=f"{save_loc}{name}step/train/")
    writer_va_st = SummaryWriter(log_dir=f"{save_loc}{name}step/test/")

    n_latlon = 64 * 128
    n_window = 1440
    N_tr = n_latlon * n_window
    N_va = (n_latlon * n_window) // portion + 1
    patience = N_tr // bs * patience
    save_name = save_loc + name + "ae.h5"

    # Counters
    lastbatch_tr = N_tr // bs + 1
    lastbatch_va = N_va // bs + 1
    val_batches = lastbatch_va if val_batches == 0 else val_batches
    epoch = init_epoch[0] if isinstance(init_epoch, list) else init_epoch
    pass_tr = epoch % lastbatch_tr
    pass_va = (epoch // num_steps) * val_batches % lastbatch_va
    baseline_loss = 0
    trn = {}
    lbvl = epoch if lbvl == 0 else lbvl  # epoch of last best validation loss
    y_tr = 0
    info(f"Loading test data: year {tst_yr:02d}.")
    tst = get_data(tst_yr, transform=transform)
    save_every = n_latlon * (1440 // 12)  # monthly.
    N_sample = init_epoch[1] * save_every if isinstance(init_epoch, list) else 0
    init_epoch = epoch
    while epoch <= epochs:
        # Shuffle when entire training/validation set have been passed through, and load in new trn years if necessary.
        if pass_tr == 0 or epoch == init_epoch:
            if sample_kind == "original":
                info("No sampling strategy!")
                idxinds_tr = (
                    np.random.permutation(N_tr)
                    if epoch == init_epoch
                    else np.random.permutation(idxinds_tr)
                )
            elif sample_kind == "histeq":
                functional, nbins, t, maxrepeat = s_funcparams
                info(
                    f"Yes sampling strategy:nbins={nbins}, t={t}, maxrepeat={maxrepeat}"
                )
                idxinds_tr = DB_get_bins_ind_alpha(
                    functional, trn_yrs[y_tr], nbins, t, maxrepeat
                )
                N_tr = idxinds_tr.shape[0]
                lastbatch_tr = N_tr // bs + 1
                pass_tr = epoch % lastbatch_tr if pass_tr != 0 else 0
            index_info_tr = get_la_lo_tstart(idxinds_tr)
            info(f"Loading train data: year {trn_yrs[y_tr]:02d}.")
            trn = get_data(trn_yrs[y_tr], transform=transform, data=trn)
            y_tr = y_tr + 1 if y_tr < len(trn_yrs) - 1 else 0
        if pass_va == 0 or epoch == init_epoch:
            idxinds_va = (
                np.random.choice(np.arange(0, N_tr), N_va, replace=False)
                if epoch == init_epoch
                else np.random.permutation(idxinds_va)
            )
            index_info_va = get_la_lo_tstart(idxinds_va)

        idxst = pass_tr * bs
        idxend = (pass_tr + 1) * bs if pass_tr != lastbatch_tr - 1 else N_tr
        N_sample += idxend - idxst
        info(f"Batch # {epoch}, month#{N_sample//save_every}: ", end="")
        loss_tr, rel2norm_tr = train_batch(
            trn,
            [idxst, idxend],
            index_info_tr,
            batch_alloc,
            model,
            loss_fn,
            optimizer,
            device,
            get_batch,
            l1_amp=l1_amp,
        )
        writer_tr_sa.add_scalar("Loss", loss_tr, N_sample // save_every)
        writer_tr_sa.flush()
        writer_tr_sa.add_scalar("Rel2Norm", rel2norm_tr, N_sample // save_every)
        writer_tr_sa.flush()
        writer_tr_st.add_scalar("Loss", loss_tr, epoch)
        writer_tr_st.flush()
        writer_tr_st.add_scalar("Rel2Norm", rel2norm_tr, epoch)
        writer_tr_st.flush()
        pass_tr += 1
        if epoch % num_steps == 0 or epoch == init_epoch:
            cum_loss = 0
            cum_rel2norm = 0
            tic = perf_counter()
            for _ in range(val_batches):
                idxst = pass_va * bs
                idxend = (pass_va + 1) * bs if pass_va != lastbatch_va - 1 else N_va
                l, r = test_batch(
                    tst,
                    [idxst, idxend],
                    index_info_va,
                    batch_alloc,
                    model,
                    loss_fn,
                    device,
                    get_batch,
                )
                cum_loss += l
                cum_rel2norm += r
                pass_va += 1
                if pass_va == lastbatch_va:
                    if val_batches != lastbatch_va:
                        info("Completed a pass through validation set!")
                    pass_va = 0
            validation_loss = cum_loss / val_batches
            rel2norm_va = cum_rel2norm / val_batches
            toc = perf_counter()
            info(
                f", Validation loss : {validation_loss:>7e}, rel2norm:{rel2norm_va:>7e}, {toc-tic:>5f} (sec)."
            )
            writer_va_sa.add_scalar("Loss", validation_loss, N_sample // save_every)
            writer_va_sa.flush()
            writer_va_sa.add_scalar("Rel2Norm", rel2norm_va, N_sample // save_every)
            writer_va_sa.flush()
            writer_va_st.add_scalar("Loss", validation_loss, epoch)
            writer_va_st.flush()
            writer_va_st.add_scalar("Rel2Norm", rel2norm_va, epoch)
            writer_va_st.flush()
            if epoch == init_epoch or baseline_loss == 0:
                baseline_loss = rel2norm_va
                info("Saving initial version of this training segment.")
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    h5str,
                )
            elif baseline_loss > rel2norm_va:
                lbvl = epoch
                info(
                    f"Validation relative norm has improved from {baseline_loss:1.4e} to {rel2norm_va:1.4e} and therefore saving the model."
                )
                baseline_loss = rel2norm_va  # validation_loss
                h5str = (
                    f"{save_name}e{epoch:04}_r2n{rel2norm_va:1.4e}.h5"
                    if save_versions is True
                    else save_name
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    h5str,
                )
        else:
            info(".")
        if pass_tr == lastbatch_tr:
            info("Completed a pass through training set!")
            pass_tr = 0
        if epoch - lbvl > patience:
            info(f"Early stop: Model has not improved since {lbvl}.")
            return epoch
        epoch += 1
    return epoch


def unstandardize(stats, data):
    """
    Un-standardize by multiplying std and adding mean back. 
    """
    for var in VAR_NAMES:
        data[var] *= stats[f"{var}_std"]
        data[var] += stats[f"{var}_mean"]
