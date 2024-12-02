

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4


def read_field(ecco_dir,vel_dir,year):

    # get the velmass file
    print('        - Reading '+vel_dir+'VELMASS_'+str(year)+'.nc')
    ds = nc4.Dataset(ecco_dir+'/'+vel_dir+'VELMASS/'+vel_dir+'VELMASS_'+str(year)+'.nc')
    if vel_dir == 'U':
        i = ds.variables['i_g'][:]
        j = ds.variables['j'][:]
    if vel_dir == 'V':
        i = ds.variables['i'][:]
        j = ds.variables['j_g'][:]
    k = ds.variables['k'][:]
    tile = ds.variables['tile'][:]
    time = ds.variables['time'][:]
    timestep = ds.variables['timestep'][:]
    velmass = ds.variables[vel_dir+'VELMASS'][:,:,:,:,:]
    ds.close()

    # get the ETAN field
    print('        - Reading ETAN_' + str(year) + '.nc')
    ds = nc4.Dataset(ecco_dir+'/ETAN/ETAN_'+str(year)+'.nc')
    etan = ds.variables['ETAN'][:, :, :, :]
    ds.close()

    print('        - Reading the hFac and depth grids')
    hFac = np.zeros((len(k), len(tile), len(i), len(j)))
    depth = np.zeros((len(tile), len(i), len(j)))
    for tile_number in range(len(tile)):
        # get the HFAC field
        ds = nc4.Dataset(ecco_dir + '/Grid/GRID.' + '{:04d}'.format(tile_number+1) + '.nc')
        if vel_dir=='U':
            hFac_tile = ds.variables['hFacW'][:, :, :]
        if vel_dir=='V':
            hFac_tile = ds.variables['hFacS'][:, :, :]
        depth_tile = ds.variables['Depth'][:,:]
        ds.close()
        hFac[:,tile_number,:,:] = hFac_tile
        depth[tile_number,:,:] = depth_tile


    full_set = {'i':i,'j':j,'k':k,'tile':tile,'time':time,
                'timestep':timestep,'velmass':velmass,'etan':etan,'hFac':hFac,'depth':depth}

    return(full_set)

def convert_velmass_to_vel(full_set):
    velmass = full_set['velmass']
    etan = full_set['etan']
    hFac = full_set['hFac']
    depth = full_set['depth']

    # see here: https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Volume_budget_closure.html

    print('          - Calculating conversions... ')
    vel = np.zeros_like(velmass)
    for timestep in range(np.shape(velmass)[0]):
        for tile in range(np.shape(velmass)[2]):
            s_star = 1 + etan[timestep,tile,:,:]/depth[tile,:,:]
            # print('s_star:',np.min(s_star),np.mean(s_star),np.max(s_star))
            # print(np.shape(velmass[timestep,:,tile,:,:]),np.shape(hFac[:,tile,:,:]),np.shape(s_star))
            vel[timestep,:,tile,:,:] = velmass[timestep,:,tile,:,:] * hFac[:,tile,:,:] * s_star

    # double check its masked?
    print('          - Double checking masking is correct')
    for timestep in range(np.shape(velmass)[0]):
        for tile in range(np.shape(velmass)[2]):
            subset = vel[timestep,:,tile,:,:]
            subset[hFac[:,tile,:,:]==0]=0
            vel[timestep,:,tile,:,:] = subset

    # vel[hFac==0] = 0
    vel[np.isnan(vel)] = 0

    return(vel)

def write_field(ecco_dir,vel_dir,year,full_set,vel):

    # get the velmass file
    ds = nc4.Dataset(ecco_dir+'/'+vel_dir+'VEL/'+vel_dir+'VEL_'+str(year)+'.nc','w')

    if vel_dir == 'U':
        ds.createDimension('i_g', np.size(full_set['i']))
        ds.createDimension('j', np.size(full_set['j']))
    if vel_dir == 'V':
        ds.createDimension('i', np.size(full_set['i']))
        ds.createDimension('j_g', np.size(full_set['j']))
    ds.createDimension('k', np.size(full_set['k']))
    ds.createDimension('tile', np.size(full_set['tile']))
    ds.createDimension('time', np.size(full_set['time']))

    if vel_dir == 'U':
        var = ds.createVariable('i_g', 'f4', ('i_g', ))
        var[:] = full_set['i']
        var = ds.createVariable('j', 'f4', ('j',))
        var[:] = full_set['j']
    if vel_dir == 'V':
        var = ds.createVariable('i', 'f4', ('i', ))
        var[:] = full_set['i']
        var = ds.createVariable('j_g', 'f4', ('j_g',))
        var[:] = full_set['j']

    var = ds.createVariable('k', 'f4', ('k', ))
    var[:] = full_set['k']

    var = ds.createVariable('tile', 'f4', ('tile',))
    var[:] = full_set['tile']

    var = ds.createVariable('time', 'f4', ('time',))
    var[:] = full_set['time']
    
    var = ds.createVariable('timestep', 'f4', ('time',))
    var[:] = full_set['timestep']

    if vel_dir=='U':
        var = ds.createVariable(vel_dir+'VEL', 'f4', ('time', 'k', 'tile', 'i_g', 'j'))
        var[:, :, :, :, :] = vel
    if vel_dir=='V':
        var = ds.createVariable(vel_dir+'VEL', 'f4', ('time', 'k', 'tile', 'i', 'j_g'))
        var[:, :, :, :, :] = vel

    ds.close()



ecco_dir = '/media/basil/Elements/data/llc270'


for vel_dir in ['U','V']:
    for year in range(2017,2018):

        print('  - Reading the full set')
        full_set = read_field(ecco_dir,vel_dir,year)

        print('  - Converting values')
        vel = convert_velmass_to_vel(full_set)

        print('  - Writing output file')
        write_field(ecco_dir,vel_dir,year,full_set,vel)


