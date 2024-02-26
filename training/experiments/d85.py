name='d85/'; 
n_d=[128,64,32,1]; n_in=40; n_out=33
h5str='/home/lmy7879/gwp/h5files/trn_stats08.h5'; stdim=0;
from utils08 import get_batch, get_data, weightedLoss
bs = 10_000
lr = 1e-4; l1_amp=1e-10; l2_amp=1e-9
functional, nbins, t, maxrepeat = "urange", 100, 0.10, 100

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
train_loop5([trn_yrs, tst_yr],model, loss_fn, optimizer, scheduler, metrics, train_info, funcs , sampling)
