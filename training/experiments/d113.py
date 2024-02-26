name='d113/'; arch_type='dnn'
n_d = [184,92,46,1]; n_in=40; n_out=33
h5str='/home/lmy7879/gwp/h5files/trn_stats08.h5'; stdim=0;
from utils08 import get_batch, get_data; weightedLoss=False
bs = 10_000
lr = 1e-4; l1_amp=1e-10; l2_amp=1e-9
functional, nbins, t, maxrepeat = "urange", 100, 0.15, 100

from utils import *
import sys
from importlib import reload

# Configurations
save_loc='/scratch/lmy7879/mweights/'
if os.path.isdir(save_loc+name) is False:
    os.mkdir(save_loc+name)
save_versions=False
# Check cpu or gpu.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

# Load architecture
sys.path.append('architectures')
import d_model10 as m;
model=m.model(n_d, n_in, n_out).to(device);
for branch in model.branches: branch.to(device) 
print(f"This model has {count_params(model):d} parameters.")
fid = h5py.File(save_loc+name+"description.h5","w")
fid["n_d"]=n_d;
fid["n_in"]=n_in;   fid["n_out"]=n_out;
fid["model_architecture"]="d_model10"; fid.close();
print("Saved experiment configurations.")


# Load data
trn_yrs = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]; tst_yr = 60
stats = save_ao_getstats(h5str, stdim, trn_yrs)
transform=partial(standardize,stats)
# data = get_data(trn_yrs[0],transform=transform)

# # Test get_batch. and model working on batch.
# idxst=1; idxend=10; idxinds = np.array(range(10))
# index_info = get_la_lo_tstart(idxinds, n_mem=1)
# bs = idxend-idxst+1
# bs=10
# batch_alloc = [torch.zeros(bs,2,40), torch.zeros(bs), torch.zeros(bs,40), torch.zeros(bs,40)]
# x3, xloc, gU, pgU = batch_alloc
# x3, xloc, gU, pgU = get_batch(data, idxst, idxend, index_info, batch_alloc)
# x3, xloc, gU, pgU = x3.to(device), xloc.to(device), gU.to(device), pgU.to(device)
# pgU = model([x3,pgU])
 
checkpoint = None 
loss_fn=nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_amp)
if checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['weights'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0, verbose=True)

num_steps = 100
val_batches = 0 
epochs = 1_000_000
patience= 50 # wait 25 passes through the training set for early stopping.
portion = 20
init_epoch = 0

batch_alloc = [torch.zeros(bs,2,40), torch.zeros(bs), torch.zeros(bs,40), torch.zeros(bs,40)]
train_info = [init_epoch, bs, num_steps, val_batches, epochs, save_loc, name, patience, portion, device, l1_amp, save_versions, transform, batch_alloc, weightedLoss]
funcs = [get_batch, get_data]
print("Start training!!")
metrics=None
sampling = ["histeq", [functional,nbins,t,maxrepeat]]

# del data
train_loop5([trn_yrs, tst_yr],model, loss_fn, optimizer, scheduler, metrics, train_info, funcs, sampling)


# Load best model
best_model = torch.load(f"/scratch/lmy7879/mweights/{name:s}{arch_type:s}.h5")
model.load_state_dict(best_model['weights'])


# Compute errors
# import utils08 as u6
stats0 = save_ao_getstats(h5str, 0, trn_yrs)
stats1= save_ao_getstats(h5str, 1, trn_yrs)
# Set batch size and num_steps (# of steps before validation.)
bs = 50_000
N_tr = M; N_va = N_tr;
# Load data.
batch_alloc = [torch.zeros(bs,2,40), torch.zeros(bs), torch.zeros(bs,40), torch.zeros(bs,40)]
lastbatch_va = N_va//bs+1; val_batches = lastbatch_va; # printL = False
idxinds_va=np.array(range(N_tr))
la,lo,tstart=get_la_lo_tstart(idxinds_va)
errors = ['absolute_component_errors', 'relative_component_errors','standardized_component_errors','absolute_norm_errors','2norms', 'relative_norm_errors']
load2mem=True; transform_cc=None
batch_info = [lastbatch_va, bs, N_va, get_batch, [la,lo,tstart], batch_alloc, transform_cc, load2mem, device]
tst_yrs2 = [21,22,23,24,25,26,27,28,29,30,56,5,58,59,60];
for tst_yr in tst_yrs2:
    tst = get_data(tst_yr,transform=transform)
    save_loc = f"/scratch/lmy7879/data2analyze/year{tst_yr:d}errs/{name[:-1]:s}_errs.h5"
    evecs= compute_errs2(tst, [stats0,stats1], batch_info, model, errors, save_type=save_loc)
    del tst


