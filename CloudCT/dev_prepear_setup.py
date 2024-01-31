import at3d
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import scipy.io as sio
import os
import logging
from collections import OrderedDict
import xarray as xr
from CloudCTUtils import *



"""
The default sensor generator:

The basic sensor generator (pyshdom.sensor.make_sensor_dataset)takes 
np.array's of the angles and positions of pixels. It makes no 
assumptions about the FOV of the pixels. A possible default 
behaviour is to assume that each pixel is modeled by a single 
infinitesmal ray at the pixel's location and pointing direction. 
This default behaviour can be chosen by selecting 
fill_ray_variables=True. Note that ray variables must be defined in 
the sensor for it to be valid. So you must use this option or 
generate the rays from the pixels according to a model.
"""

mu = phi = x = y = z = np.array([1.0])
stokes = ['I']
wavelength = 0.86

sensor = at3d.sensor.make_sensor_dataset(x, y, z, \
                                            mu, phi, stokes, \
                                            wavelength, \
                                            fill_ray_variables=True)
at3d.checks.check_sensor(sensor)


"""
The Perspective Sensor:

The perspective sensor generator (at3d.sensor.perspective_projection) 
is a pinhole sensor and is defined by its location, pointing direction 
and resolution. This generator also includes explicit method for 
generating sub-pixel rays to model the FOV.

Note that the attributes contain the input information used to
generate this sensor.
"""

sensor_dict = at3d.containers.SensorsDict()

"""
SensorDict
Groups of sensors which share an uncertainty model are designated as instruments. 
And groups of instruments are stored in pyshdom.containers.SensorsDict, which is a glorified OrderedDict.
These groups of sensors are labeled. The pyshdom.containers.SensorsDict is the container which interacts 
with the other high level objects. Individual datasets that contain pixel geometries can still be used to 
directly interface with
a pyshdom.solver.RTE object to obtain observables at the specified sensor geometry.

Currently, I want to avoid using sensors names in nither sensor nor SensorDict.
I will follow the sensor_index es, measn, link a name to sensor_index.
"""


#---------------------------------------
#--------------------------------------------
#--------------------------------------------


"""
Prepare CloudCT simple string of pearls setup:
10 satellites with 100km distance (on orbit arc).

"""
# orbit altitude:
Rsat = 500 # km
GSD = 0.5*0.02 # in km, it is the ground spatial resolution.
wavelengths_micron = 0.672  #0.672 , 1.6
sun_azimuth = 45
sun_zenith = 155
maxiter = 150
n_jobs = 60
SATS_NUMBER_SETUP = 10 # satellites number to build the setup, for the inverse, we can use less satellites.

#--------------------------------------------
#--------------------------------------------
#--------------------------------------------

# load a cloud for loading the domain prperties:
IFVISUALIZE = True
#CloudFieldFile = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/BOMEX_21600_515x515x71_7286.txt"
CloudFieldFile = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/BOMEX_57600_58x65x42_7844.txt"
cloud_scatterer = at3d.util.load_from_csv(CloudFieldFile, density='lwc')

# find reff/veff min/max:
a=np.array(cloud_scatterer['reff'])
b = a[a>0]
reff_min = min(b.min() - 0.3, 1)
reff_max = b.max()

a=np.array(cloud_scatterer['veff'])
b = a[a>0]
veff_min = b.min()
veff_max = b.max() + 0.01

# The Nones make problems in the pyshdom.grid.resample_onto_grid()
# So I get rid of them.
cloud_scatterer['density'] = (['x', 'y', 'z'], np.nan_to_num(cloud_scatterer.density))
cloud_scatterer['reff'] = (['x', 'y', 'z'], np.nan_to_num(cloud_scatterer.reff))
cloud_scatterer['veff'] = (['x', 'y', 'z'], np.nan_to_num(cloud_scatterer.veff))

xgrid = cloud_scatterer.x
ygrid = cloud_scatterer.y
zgrid = cloud_scatterer.z

dx = cloud_scatterer.delx.item()
dy = cloud_scatterer.dely.item()
dz = round(np.diff(zgrid)[0],5) 
nx, ny, nz = cloud_scatterer.dims['x'],cloud_scatterer.dims['y'],cloud_scatterer.dims['z']

# Visualize LWC data to validate the load:
show_scatterer(cloud_scatterer)
#-----------------------------------------------
#USED FOV, RESOLUTION and SAT_LOOKATS:
PIXEL_FOOTPRINT = GSD # km
L = max(xgrid.data.max() + dx, ygrid.data.max() + dy)

fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
cny = int(np.floor(L/PIXEL_FOOTPRINT))
cnx = int(np.floor(L/PIXEL_FOOTPRINT))

CENTER_OF_MEDIUM_BOTTOM = [0.5*nx*dx , 0.5*ny*dy , 0]
# Somtimes it is more convinent to use wide fov to see the whole cloud
# from all the view points. so the FOV is aslo tuned:
IFTUNE_CAM = True
# --- TUNE FOV, CNY,CNX:
if(IFTUNE_CAM):
    L = 1.2*L
    fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
    cny = int(np.floor(L/PIXEL_FOOTPRINT))
    cnx = int(np.floor(L/PIXEL_FOOTPRINT))    

# not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
# tuning is applied by the variavle LOOKAT.
LOOKAT = CENTER_OF_MEDIUM_BOTTOM
if(IFTUNE_CAM):
    LOOKAT[2] = 0.68*nx*dz # tuning. if IFTUNE_CAM = False, just lookat the bottom

SAT_LOOKATS = np.array(SATS_NUMBER_SETUP*LOOKAT).reshape(-1,3)# currently, all satellites lookat the same point.

print(20*"-")
print(20*"-")
print(20*"-")

print("CAMERA intrinsics summary")
print("fov = {}[deg], cnx = {}[pixels],cny ={}[pixels]".format(fov,cnx,cny))

print(20*"-")
print(20*"-")
print(20*"-")

sat_positions, near_nadir_view_index, theta_max, theta_min = \
    StringOfPearls(SATS_NUMBER = SATS_NUMBER_SETUP,\
    orbit_altitude = Rsat,\
    move_nadir_x=CENTER_OF_MEDIUM_BOTTOM[0],\
    move_nadir_y=CENTER_OF_MEDIUM_BOTTOM[1])

names = ["sat"+str(i+1) for i in range(len(sat_positions))] 
# we intentialy, work with projections lists.
up_list = np.array(len(sat_positions)*[0,1,0]).reshape(-1,3) # default up vector per camera.

for position_vector,lookat_vector,up_vector,name in zip(sat_positions,\
                              SAT_LOOKATS,up_list,names):

    loop_sensor = at3d.sensor.perspective_projection(wavelength = wavelengths_micron, fov = fov,\
    x_resolution = cnx, y_resolution = cny,\
    position_vector = position_vector, lookat_vector = lookat_vector,\
    up_vector = up_vector, stokes=['I'], sub_pixel_ray_args={'method':at3d.sensor.stochastic,'nrays':1})


    sensor_dict.add_sensor('CloudCT', loop_sensor)
    
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------





sensor_list = sensor_dict['CloudCT']['sensor_list']


at3d.sensor.show_sensors(sensor_list, scale = 50, axisWidth = 1.0, axisLenght=30, Show_Rays =  True, FullCone = True)
mlab.orientation_axes()

mlab.show()

if(1):
    print('start Simulating Radiances')
    """
    Simulating Radiances
    """
    #load atmosphere
    atmosphere = xr.open_dataset('../data/ancillary/AFGL_summer_mid_lat.nc')
    #subset the atmosphere, choose only the bottom four km.
    reduced_atmosphere = atmosphere.sel({'z': atmosphere.coords['z'].data[atmosphere.coords['z'].data <= 4.0]})
    #merge the atmosphere and cloud z coordinates
    merged_z_coordinate = at3d.grid.combine_z_coordinates([reduced_atmosphere,cloud_scatterer])    
    
    # -----------------------------------------------------
    # make a grid for microphysics which is just the cloud grid.
    rte_grid = at3d.grid.make_grid(dx,cloud_scatterer.x.data.size,
                                   dy,cloud_scatterer.y.data.size,
                              cloud_scatterer.z)

    cloud_scatterer_on_rte_grid = at3d.grid.resample_onto_grid(rte_grid, cloud_scatterer)

    # We choose a gamma size distribution and therefore need to define a 'veff' variable.
    size_distribution_function = at3d.size_distribution.gamma
    
    wavelengths = sensor_dict.get_unique_solvers()
    wavelength_band = (wavelengths[0], wavelengths[0])
    
    wavelen1, wavelen2 = wavelength_band
    wavelength_averaging = False
    formatstr = 'TEST_Water_{}nm.nc'.format(int(1e3*wavelength_band[0]))
    safe_mkdirs('../mie_tables')
    if not (wavelen1 == wavelen2):
        wavelength_averaging = True   
        formatstr = 'TEST_averaged_Water_{}-{}nm.nc'.format(int(1e3*wavelength_band[0]), int(1e3*wavelength_band[1]))
    mono_path = os.path.join('../mie_tables', formatstr)

    
    # Exact OpticalPropertyGenerator:
    # get_mono_table will first search a directory to see if the requested table exists otherwise it will calculate it. 
    # You can save it to see if it works.
    mie_mono_table = at3d.mie.get_mono_table(
        'Water',wavelength_band,
        max_integration_radius=65.0,
        minimum_effective_radius=0.1,
        relative_dir='../mie_tables',
        verbose=False
    )
    
    mie_mono_table.to_netcdf(mono_path)
    mie_mono_tables = OrderedDict()
    mie_mono_tables[wavelength_band[0]] = mie_mono_table
    
    optical_prop_gen = at3d.medium.OpticalPropertyGenerator(
        'cloud',
        mie_mono_tables, 
        size_distribution_function,
        particle_density=1.0, 
        maxnphase=None,
        interpolation_mode='exact',
        density_normalization='density',#The density_normalization argument is a convenient
        reff=np.linspace(reff_min-0.3,30.0,50),
        veff=np.linspace(0.01,veff_max,15)
    )
    
    optical_properties = optical_prop_gen(cloud_scatterer_on_rte_grid)
    # The optical properties produced by this contain all of the information 
    # required for the RTE solver. Note that the attributes track the inputs
    # and the microphysical properties are also brought along for traceability
    # purposes.
    
    # If you generate your own optical properties they must pass this check to be used in the solver.
    at3d.checks.check_optical_properties(optical_properties[wavelength_band[0]])
    
    # one function to generate rayleigh scattering.
    rayleigh_scattering = at3d.rayleigh.to_grid(wavelengths,atmosphere,rte_grid)    
    
    """
    Define Solvers:
    Define solvers last based on the sensor's spectral information.


    """
    
    solvers_dict = at3d.containers.SolversDict()
    # note we could set solver dependent surfaces / sources / numerical_config here
    # just as we have got solver dependent optical properties.
    
    for wavelength in wavelengths:
        medium = {
            'cloud': optical_properties[wavelength],
            'rayleigh':rayleigh_scattering[wavelength]
         }
        config = at3d.configuration.get_config()
        solvers_dict.add_solver(
            wavelength,
            at3d.solver.RTE(
                numerical_params=config,
                surface=at3d.surface.lambertian(0.05),
                source=at3d.source.solar(wavelength, 0.5,0.0),
                medium=medium,
                num_stokes=1#sensor_dict.get_minimum_stokes()[wavelength],
            )                   
   )
        
        
    # solve the 4 RTEs in parallel AND get the measurements.
    sensor_dict.get_measurements(solvers_dict, n_jobs=n_jobs, verbose=True)
    
    # see images:
    for instrument in sensor_dict:
        sensor_images = sensor_dict.get_images(instrument)
        for sensor in sensor_images:
            plt.figure()
            sensor.I.T.plot()
            plt.title(instrument)    
    
    plt.show()
    print('Done Simulating Radiances')