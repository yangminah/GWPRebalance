# import h5py
# from utils import load_year
# import numpy as np
# fid = h5py.File('/scratch/lmy7879/data2analyze/urange.h5','w')
# for yr in [21, 22, 23, 24, 25, 26 ,27, 28, 29, 30]:
#     print(yr)
#     fidyr = load_year(yr)
#     fid[f"{yr:02d}"]=np.amax(fidyr['ucomp'][()],1)-np.amin(fidyr['ucomp'][()],1)
#     fidyr.close()
# fid.close()


import netCDF4
import numpy as np
fid = netCDF4.Dataset('/scratch/lmy7879/data2analyze/year60_80levels.nc','a')

''' The 80 levels comprise of: 
[0::2] = [0, 2, ..., 76, 78] pfull,
[1::2] = [1, 3, ..., 77, 79] phalf without the top lid (since it is set to 0).
'''
u = fid['ucomp'][()]
h = fid['hght'][()]

# calculate shear at all levels. 
shear = np.zeros(u.shape)
shear[:,1:78+1,:,:] = u[:,2:,:,:].copy()
shear[:,1:78+1,:,:] -= u[:,0:77+1,:,:]
shear[:,0,:,:] = u[:,1,:,:].copy() - u[:,0,:,:].copy()
shear[:,-1,:,:] = u[:,-2,:,:].copy() - u[:,-1,:,:].copy()
del u
hdiff = h[:,2:,:,:].copy()
hdiff -= h[:,0:77+1,:,:]
shear[:,1:78+1,:,:] /= hdiff
del hdiff
shear[:,0,:,:] /= (h[:,1,:,:].copy()-h[:,0,:,:].copy())
shear[:,-1,:,:]/= (h[:,-2,:,:].copy()-h[:,-1,:,:].copy())

pfull_s = fid.createDimension('pfull_short', 40)
fid.createVariable('pfull_short',np.float32,'pfull_short')
fid['pfull_short'][:]=fid['pfull'][0::2]
# shear_fid = fid.createVariable('shear',np.float32,('time','pfull_short','lat','lon'))
shear_fid = fid.createVariable('shear',np.float32,('time','pfull','lat','lon'))
shear_fid.units = '1/s'
shear_fid.long_name='shear'
shear_fid[:,:,:,:]=shear


# compute source levels.
# First get 315 hPa. 
source_level_pressure=315
lat = fid['lat'][()]/180*np.pi
pfull = fid['pfull']
klevel_of_source=0
for (k,pf) in enumerate(pfull):
    if pf > source_level_pressure:
        klevel_of_source=k
        break
source_level = np.zeros(lat.shape).astype(int)
kmax = 79
for (j,la) in enumerate(lat):
    source_level[j] = (kmax+1)-((kmax+1-klevel_of_source)*np.cos(lat[j])+0.5)


int_abs_shear = np.zeros((1440,40,64,128))
# For every latitude,
for j in range(lat.shape[0]):
    sl = source_level[j]
    print(j)
    # compute the integrated absolute shear from just above the source level all the way to the model top. 
    for lstar in range(0,sl,2):
        #print(lstar)
        # Initialize the quadrature from source level,
        int_abs_shear[:,lstar//2,j,:] = .5*np.sum(np.abs(shear[:,sl:,j,:]),axis=1)*(h[:,sl,j,:]-h[:,sl+1,j,:])
        if lstar < sl:
            # then go up a level until lstar. 
            for l in range(sl-1,lstar,-1):
                int_abs_shear[:,lstar//2,j,:] += .5*(np.abs(shear[:,l,j,:])+np.abs(shear[:,l+1,j,:]))*(h[:,l,j,:]-h[:,l+1,j,:])


int_abs_shear_fid = fid.createVariable('int_abs_shear',np.float32,('time','pfull_short','lat','lon'))
int_abs_shear_fid.units = '1/s'
int_abs_shear_fid.long_name='int_abs_shear'
int_abs_shear_fid[:,:,:,:]=int_abs_shear

del int_abs_shear
del shear
u = fid['ucomp'][()]
u_range = np.zeros((1440,40,64,128))
# For each latitude,
for j in range(lat.shape[0]):
    sl = source_level[j]
    # starting from the source level all the way to the model top,
    for l in range(sl//2-1,-1,-1):
        # compute the maximum zonal wind range from level l to the source level. 
        #u_range[:,l,j,:] =  np.max(u[:,2*l::2,j,:],axis=1) -  np.min(u[:,2*l::2,j,:],axis=1)
        u_range[:,l,j,:] =  np.max(u[:,2*l:sl,j,:],axis=1) -  np.min(u[:,2*l:sl,j,:],axis=1)

u_range_fid = fid.createVariable('u_range',np.float32,('time','pfull_short','lat','lon'))
u_range_fid.units = 'm'
u_range_fid.long_name='u_range'
u_range_fid[:,:,:,:]=u_range







# shear=fid['shear'][()]
# Compute only from a set source level and up. 
# int_abs_shear = np.zeros((1440,40,64,128))
# sl = 60 # source level index
# for lstar in range(0,sl,2):
#     print(lstar)
#     int_abs_shear[:,lstar//2,:,:] = .5*np.sum(np.abs(shear[:,sl:,:,:]),axis=1)*(h[:,sl,:,:]-h[:,sl+1,:,:])
#     if lstar < sl:
#         for l in range(sl-1,lstar,-1):
#             int_abs_shear[:,lstar//2,:,:] += .5*(np.abs(shear[:,l,:,:])+np.abs(shear[:,l+1,:,:]))*(h[:,l,:,:]-h[:,l+1,:,:])


# int_abs_shear_fid = fid.createVariable('int_abs_shear',np.float32,('time','pfull_short','lat','lon'))
# int_abs_shear_fid.units = '1/s'
# int_abs_shear_fid.long_name='int_abs_shear'
# int_abs_shear_fid[:,:,:,:]=int_abs_shear

# del int_abs_shear
# del shear
# u = fid['ucomp'][()]
# u_range = np.zeros((1440,40,64,128))
# for l in range(sl//2-1,-1,-1):
#     u_range[:,l,:,:] =  np.max(u[:,2*l::2,:,:],axis=1) -  np.min(u[:,2*l::2,:,:],axis=1)

# u_range_fid = fid.createVariable('u_range',np.float32,('time','pfull_short','lat','lon'))
# u_range_fid.units = 'm'
# u_range_fid.long_name='u_range'
# u_range_fid[:,:,:,:]=u_range


# time = 30; level=3; lat= 39; lon = 29

# np.max(u[time,l:,lat,lon],axis=1) -  np.min(u[time,l:,lat,lon],axis=1)
# int_abs_shear[time,level,lat,lon]
# u = fid['ucomp'][()]
# h = fid['hght'][()]
# import matplotlib
# #matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# time = 30; level=3; lat= 39; lon = 29
# plt.plot(u[time, 0::2, lat, lon], h[time, 0::2, lat, lon], label='zonal wind (m/s)')
# plt.plot(3600*shear[time, :, lat, lon], h[time, 0::2, lat, lon], label='shear (1/hour) ')
# plt.legend()
# plt.show()
