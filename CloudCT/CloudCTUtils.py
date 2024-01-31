import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import at3d
import matplotlib.pyplot as plt
# import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
import os
import logging
from collections import OrderedDict
import xarray as xr
# import transformations as transf
import pickle
import pandas as pd
import warnings
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
import copy

# -------------------------------------------------------------------------------
# ----------------------CONSTANTS------------------------------------------
# -------------------------------------------------------------------------------
r_earth = 6371.0  # km
origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def float_round(x):
    """Round a float or np.float32 to a 3 digits float"""
    if type(x) == np.float32:
        x = x.item()
    return round(x, 3)


def safe_mkdirs(path):
    """Safely create path, warn in case of race."""

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            import errno
            if e.errno == errno.EEXIST:
                warnings.warn(
                    "Failed creating path: {path}, probably a race".format(path=path)
                )


def save_to_csv(cloud_scatterer, file_name, comment_line='', OLDPYSHDOM=False):
    """
    
    A utility function to save a microphysical medium.
    After implementation put as a function in util.py under the name 
    save_to_csv.
    
    Format:
    
    
    Parameters
    ----------
    path: str
        Path to file.
    comment_line: str, optional
        A comment line describing the file.
    OLDPYSHDOM: boll, if it is True, save the txt in old version of pyshdom.
    
    Notes
    -----
    CSV format is as follows:
    # comment line (description)
    nx,ny,nz # nx,ny,nz
    dx,dy # dx,dy [km, km]
    z_levels[0]     z_levels[1] ...  z_levels[nz-1] 
    x,y,z,lwc,reff,veff
    ix,iy,iz,lwc[ix, iy, iz],reff[ix, iy, iz],veff[ix, iy, iz]
    .
    .
    .
    ix,iy,iz,lwc[ix, iy, iz],reff[ix, iy, iz],veff[ix, iy, iz]
    
    
    
    """
    xgrid = cloud_scatterer.x
    ygrid = cloud_scatterer.y
    zgrid = cloud_scatterer.z

    dx = cloud_scatterer.delx.item()
    dy = cloud_scatterer.dely.item()
    dz = round(np.diff(zgrid)[0], 5)

    REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.lwc)
    REGULAR_REFF_DATA = np.nan_to_num(cloud_scatterer.reff)
    REGULAR_VEFF_DATA = np.nan_to_num(cloud_scatterer.veff)

    y, x, z = np.meshgrid(range(cloud_scatterer.sizes.get('y')), \
                          range(cloud_scatterer.sizes.get('x')), \
                          range(cloud_scatterer.sizes.get('z')))

    if not OLDPYSHDOM:

        with open(file_name, 'w') as f:
            f.write(comment_line + "\n")
            # nx,ny,nz # nx,ny,nz
            f.write('{}, {}, {} '.format(int(cloud_scatterer.sizes.get('x')), \
                                         int(cloud_scatterer.sizes.get('y')), \
                                         int(cloud_scatterer.sizes.get('z')), \
                                         ) + "# nx,ny,nz\n")
            # dx,dy # dx,dy [km, km]
            f.write('{:2.3f}, {:2.3f} '.format(dx, dy) + "# dx,dy [km, km]\n")

            # z_levels[0]     z_levels[1] ...  z_levels[nz-1] 

            np.savetxt(f, \
                       X=np.array(zgrid).reshape(1, -1), \
                       fmt='%2.3f', delimiter=', ', newline='')
            f.write(" # altitude levels [km]\n")
            f.write("x,y,z,lwc,reff,veff\n")

            data = np.vstack((x.ravel(), y.ravel(), z.ravel(), \
                              REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
            # Delete unnecessary rows e.g. zeros in lwc
            mask = REGULAR_LWC_DATA.ravel() > 0
            data = data[mask, ...]
            np.savetxt(f, X=data, fmt='%d ,%d ,%d ,%.5f ,%.3f ,%.5f')

    else:
        # save in the old version:
        with open(file_name, 'w') as f:
            f.write(comment_line + "\n")
            # nx,ny,nz # nx,ny,nz
            f.write('{} {} {} '.format(int(cloud_scatterer.sizes.get('x')), \
                                       int(cloud_scatterer.sizes.get('y')), \
                                       int(cloud_scatterer.sizes.get('z')), \
                                       ) + "\n")

            # dx,dy ,z
            np.savetxt(f, X=np.concatenate((np.array([dx, dy]), zgrid)).reshape(1, -1), fmt='%2.3f')
            # z_levels[0]     z_levels[1] ...  z_levels[nz-1] 

            data = np.vstack((x.ravel(), y.ravel(), z.ravel(), \
                              REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
            # Delete unnecessary rows e.g. zeros in lwc
            mask = REGULAR_LWC_DATA.ravel() > 0
            data = data[mask, ...]
            np.savetxt(f, X=data, fmt='%d %d %d %.5f %.3f %.5f')


def load_from_csv_shdom(path, density=None, origin=(0.0, 0.0)):
    df = pd.read_csv(path, comment='#', skiprows=3, delimiter=' ')
    nx, ny, nz = np.genfromtxt(path, skip_header=1, max_rows=1, dtype=int, delimiter=' ')
    dx, dy = np.genfromtxt(path, max_rows=1, usecols=(0, 1), dtype=float, skip_header=2)
    z_grid = np.genfromtxt(path, max_rows=1, usecols=range(2, 2 + nz), dtype=float, skip_header=2)
    z = xr.DataArray(z_grid, coords=[range(nz)], dims=['z'])

    dset = at3d.grid.make_grid(dx, nx, dy, ny, z)

    for index, name in zip([3, 4, 5], ['lwc', 'reff', 'veff']):
        # initialize with np.nans so that empty data is np.nan
        variable_data = np.zeros((dset.sizes['x'], dset.sizes['y'], dset.sizes['z']))
        i = df.values[:, 0].astype(int)
        j = df.values[:, 1].astype(int)
        k = df.values[:, 2].astype(int)

        variable_data[i, j, k] = df.values[:, index]
        dset[name] = (['x', 'y', 'z'], variable_data)

    if density is not None:
        assert density in dset.data_vars, \
            "density variable: '{}' must be in the file".format(density)

        dset = dset.rename_vars({density: 'density'})
        dset.attrs['density_name'] = density

    dset.attrs['file_name'] = path

    return dset


# def show_scatterer(cloud_scatterer):
#
#     """
#     Show the scatterer in 3D with Mayavi.
#     """
#
#     ShowVolumeBox = True
#
#     xgrid = cloud_scatterer.x
#     ygrid = cloud_scatterer.y
#     zgrid = cloud_scatterer.z
#
#     dx = cloud_scatterer.delx.item()
#     dy = cloud_scatterer.dely.item()
#     dz = round(np.diff(zgrid)[0],5)
#
#     REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.density)
#     REGULAR_REFF_DATA = np.nan_to_num(cloud_scatterer.reff)
#     REGULAR_VEFF_DATA = np.nan_to_num(cloud_scatterer.veff)
#
#
#     show_field = REGULAR_LWC_DATA
#     data_type = 'LWC [g/m^3]'
#
#     mlab.figure(size=(600, 600))
#     X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
#     figh = mlab.gcf()
#     src = mlab.pipeline.scalar_field(X, Y, Z, show_field, figure=figh)
#
#     src.spacing = [dx, dy, dz]
#     src.update_image_data = True
#
#     isosurface = mlab.pipeline.iso_surface(src, contours=[0.1*show_field.max(),\
#                                                           0.2*show_field.max(),\
#                                                           0.3*show_field.max(),\
#                                                           0.4*show_field.max(),\
#                                                           0.5*show_field.max(),\
#                                                           0.6*show_field.max(),\
#                                                           0.7*show_field.max(),\
#                                                           0.8*show_field.max(),\
#                                                           0.9*show_field.max(),\
#                                                           ],opacity=0.9,figure=figh)
#
#     mlab.outline(figure=figh,color = (1, 1, 1))  # box around data axes
#     mlab.orientation_axes(figure=figh)
#     mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)")
#     color_bar = mlab.colorbar(title=data_type, orientation='vertical', nb_labels=5)
#
#     if(ShowVolumeBox):
#         # The _max is one d_ after the last point of the xgrid (|_|_|_|_|_|_|_->|).
#         x_min = xgrid[0]
#         x_max = round(xgrid[-1].item() + dx,5)
#
#         y_min = ygrid[0]
#         y_max = round(ygrid[-1].item() + dy,5)
#
#         z_min = zgrid[0]
#         z_max = round(zgrid[-1].item() + dz,5)
#
#         xm = [x_min, x_max, x_max, x_min, x_max, x_max, x_min, x_min ]
#         ym = [y_min, y_min, y_min, y_min, y_max, y_max, y_max, y_max ]
#         zm = [z_min, z_min, z_max, z_max, z_min, z_max, z_max, z_min ]
#         # Medium cube
#         triangles = [[0,1,2],[0,3,2],[1,2,5],[1,4,5],[2,5,6],[2,3,6],[4,7,6],[4,5,6],[0,3,6],[0,7,6],[0,1,4],[0,7,4]];
#         obj = mlab.triangular_mesh( xm, ym, zm, triangles,color = (0.0, 0.17, 0.72),opacity=0.3,figure=figh)
#
#
#     #mlab.show()

# ---------------------------------------------------
def StringOfPearls(SATS_NUMBER=10, orbit_altitude=500, widest_view=False, move_nadir_x=0, move_nadir_y=0):
    """
    Set orbit parmeters:
         input:
         SATS_NUMBER - int, how many satellite to put?
         move_nadir_x/y - move in x/y to fit perfect nadir view.

         WIDEST_VIEW - bool, If WIDEST_VIEW False, the setup is the original with 100km distance between satellites.
         If it is True the distance become 200km.

         returns sat_positions: np.array of shape (SATS_NUMBER,3).
         The satellites setup alwas looks like \ \ | / /.
    """
    Rsat = orbit_altitude  # km orbit altitude
    R = r_earth + Rsat
    r_orbit = R

    if (widest_view):
        Darc = 200
    else:
        Darc = 100  # km # distance between adjecent satellites (on arc).

    Dtheta = Darc / R  # from the center of the earth.

    # where to set the satelites?
    theta_config = np.arange(-0.5 * SATS_NUMBER, 0.5 * SATS_NUMBER) * Dtheta  # double for wide angles

    theta_config = theta_config[::-1]  # put sat1 to be the rigthest
    # print('Satellites angles relative to center of earth:')
    # for i,a in enumerate(theta_config):
    # print("{}: {}".format(i,a))

    theta_max, theta_min = max(theta_config), min(theta_config)

    X_config = r_orbit * np.sin(theta_config) + move_nadir_x
    Z_config = r_orbit * np.cos(theta_config) - r_earth
    Y_config = np.zeros_like(X_config) + move_nadir_y

    sat_positions = np.vstack([X_config, Y_config, Z_config])  # path.shape = (3,#sats) in km.

    Satellites_angles = np.rad2deg(np.arctan(X_config / Z_config))
    print('Satellites angles are:')
    print(Satellites_angles)
    print("max angle {}\nmin angle {}\n".format(theta_max, theta_min))

    # find near nadir view:
    # since in this setup y=0:
    near_nadir_view_index = np.argmin(np.abs(X_config))

    return sat_positions.T, near_nadir_view_index, theta_max, theta_min


def StringOfPearlsCloudBowScan(orbit_altitude=500, lookat=np.array([0, 0, 0]), cloudbow_additional_scan=6,
                               cloudbow_range=[135, 155], theta_max=60, theta_min=-60, sun_zenith=180, sun_azimuth=0):
    """
    TODO

    input:
        ORBIT_ALTITUDE - in km  , float.
        cloudbow_range - list of two elements - the cloudbow range in degrees.
        cloudbow_additional_scan - integer - how manny samples to add in the cloudbow range (input param cloudbow_range).
        theta_max, theta_min - floats - setup extrims off-nadir view angles in radian.

    """
    assert cloudbow_range[0] < cloudbow_range[1], "bad order of the cloudbow_range input"

    Rsat = orbit_altitude  # km orbit altitude
    R = r_earth + Rsat
    r_orbit = R

    sat_thetas = np.arange(theta_min, theta_max, 0.0001)
    sat_thetas = sat_thetas[::-1]  # put sat1 to be the rigthest

    X_config = r_orbit * np.sin(sat_thetas)
    Z_config = r_orbit * np.cos(sat_thetas) - r_earth
    Y_config = np.zeros_like(X_config)
    sat_positions = np.vstack([X_config, Y_config, Z_config])  # path.shape = (3,#sats) in km.
    sat_direction = lookat[:, np.newaxis] - sat_positions
    sat_direction = -sat_direction / np.linalg.norm(sat_direction, axis=0, keepdims=True)

    #### Compute sun_direction ####
    SUN_THETA = np.deg2rad(sun_zenith)
    SUN_PHI = np.deg2rad(sun_azimuth)
    # calc sun direction from lookat to sun:
    sun_x = np.sin(SUN_THETA) * np.cos(SUN_PHI)
    sun_y = np.sin(SUN_THETA) * np.sin(SUN_PHI)
    sun_z = np.cos(SUN_THETA)
    sun_direction = np.array([sun_x, sun_y, sun_z])

    # virtual_sun = lookat - 600*sun_direction
    # mlab.quiver3d(virtual_sun[0], virtual_sun[1], virtual_sun[2],
    # sun_direction[0],sun_direction[1],sun_direction[2],
    # line_width=3.0,color = (1,1,0),opacity=1,mode='2ddash',scale_factor=1)

    # Phi - scattering angle
    sat_sun_angles = np.rad2deg(np.arccos(np.dot(sun_direction, sat_direction)))
    sat_thetas = np.rad2deg(sat_thetas)  # convert to degrees
    # interpreted_thetas based on desired_phis
    desired_phis = np.linspace(start=cloudbow_range[0], stop=cloudbow_range[1], num=cloudbow_additional_scan)
    desired_thetas = np.zeros_like(desired_phis)
    # phis for angles relative to satellite and lookat.
    # thetas for angles relative to satellite and center for earth.
    j1 = np.argmin(np.abs(desired_phis[0] - sat_sun_angles))
    first_theta = sat_thetas[j1]
    d = 0.5  # degrees # TODO - find this treshold as the maximum alowed
    low_bound = first_theta - d
    up_bound = first_theta + d  # degrees
    for i, phi in enumerate(desired_phis):
        while (True):

            j = np.argmin(np.abs(phi - sat_sun_angles))
            candidat = sat_thetas[j]
            if ((low_bound <= candidat) and (candidat <= up_bound)):
                desired_thetas[i] = candidat
                low_bound = candidat - d
                up_bound = candidat + d  # degrees
                break
            else:
                sat_sun_angles[j] = -200  # give invalid value

    # result_phis should be close to desired_phis, check it here:
    test_X_config = r_orbit * np.sin(np.deg2rad(desired_thetas))
    test_Z_config = r_orbit * np.cos(np.deg2rad(desired_thetas)) - r_earth
    test_Y_config = np.zeros_like(test_X_config)
    test_sat_positions = np.vstack([test_X_config, test_Y_config, test_Z_config])  # path.shape = (3,#sats) in km.
    test_sat_direction = lookat[:, np.newaxis] - test_sat_positions
    test_sat_direction = -test_sat_direction / np.linalg.norm(test_sat_direction, axis=0, keepdims=True)
    result_phis = np.rad2deg(np.arccos(np.dot(sun_direction, test_sat_direction)))

    # assert np.all(np.abs(
    #     desired_phis - result_phis) < 2), "Something went wrong in the cloudbow scanning calculations."  # 2 degree margin
    # plt.plot(sat_thetas,sat_sun_angles)
    # plt.plot(desired_thetas,result_phis,'.')
    # plt.show()
    desired_thetas = np.deg2rad(desired_thetas)  # convert to radian
    """
    Linear interpulation is bad option here since the sat_sun_angles shape is ~parabolic.
    interpreted_thetas = np.interp(desired_phis, sat_sun_angles, sat_thetas)
    interpreted_thetas = np.deg2rad(interpreted_thetas)# convert to redian
    """

    # if we can't get all of the desired cloudbow angles, just continue scanning with the same dtheta between scans:
    dtheta = np.diff(desired_thetas)
    new_theta_inds = np.argwhere(np.abs(dtheta) < 1e-5)
    if new_theta_inds.size != 0 and np.array_equal(new_theta_inds.ravel(), np.arange(new_theta_inds[0], len(dtheta))):
        not_cloudbow_startind = int(np.argwhere(np.abs(dtheta) < 1e-4)[0])
        rest_of_dthetas = dtheta[not_cloudbow_startind - 1]
        num_of_new_thetas = len(desired_thetas) - (not_cloudbow_startind + 1)
        desired_thetas[not_cloudbow_startind + 1:] = (desired_thetas[not_cloudbow_startind] +
                                                      np.arange(1, num_of_new_thetas + 1) * rest_of_dthetas)
    elif (new_theta_inds.size != 0) or (new_theta_inds.size == 0 and np.any(np.abs(desired_phis - result_phis) < 2)):
        raise Exception("Something went wrong in the cloudbow scanning calculations.")
    # interpreted_sat_positions
    interp_X_config = r_orbit * np.sin(desired_thetas)
    interp_Z_config = r_orbit * np.cos(desired_thetas) - r_earth
    interp_Y_config = np.zeros_like(desired_thetas)
    interpreted_sat_positions = np.vstack([interp_X_config, interp_Y_config, interp_Z_config])

    print(interpreted_sat_positions)
    print("angular gap (resolution)")
    print("gaps: ", np.diff(result_phis))
    print("resolution: ", np.mean(np.diff(result_phis)))

    adjecent_distances = np.diff(interpreted_sat_positions, axis=-1)
    adjecent_distances = np.linalg.norm(adjecent_distances, axis=0)  # in km
    time_gaps = adjecent_distances / 7.612  # satellite velocity at 500km orbit is assumed to be 7.612 km/sec.
    # time_gaps in sec.
    print("time gaps between adjecent cloudbow scans are:")
    print(time_gaps)
    print("total time of cloudbow scan is:")
    print(time_gaps.sum())

    Satellites_angles = np.rad2deg(np.arctan(interp_X_config / interp_Z_config))
    print('Cloudbow satellites angles are:')
    print(Satellites_angles)

    # for each sat-center angle (theta) compute sat-sun angle (phi)

    # 1d interpolate phi --> theta

    # Arange N phis in range 135-165
    # for each phi --> theta --> x,y,z sat
    return interpreted_sat_positions.T


# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

# def apply_platform_noise(in_sensor, sigma):
#     """
#     TODO: add noisy in the position in the future.
#
#     Meanwhile it adds only orientation noise.
#
#     Parameters:
#     ------------
#     in_sensor: xr.Dataset
#         A dataset containing all of the information required to define a sensor
#         for which synthetic measurements can be simulated;
#         positions and angles of all pixels, sub-pixel rays and their associated weights,
#         and the sensor's observables. It is the input sensor. The output is out_sensor.
#
#
#     sigma: float
#         Orientation noise amplitude (std) in degrees.
#         The ralative angles roll , pitch, yaw will be sampled depending on sigma.
#
#
#     Returns
#     -------
#     out_sensor : xr.Dataset
#         An output dataset containing all of the information required to define a sensor
#         for which synthetic measurements can be simulated;
#         positions and angles of all pixels, sub-pixel rays and their associated weights,
#         and the sensor's observables.
#     """
#     assert 'Perspective' == in_sensor.attrs['projection'], "This method fits only Perspective projection."
#
#     # Sample relative rotation angles:
#     roll  =  np.deg2rad( np.random.normal(0, sigma, 1) )
#     pitch =  np.deg2rad( np.random.normal(0, sigma, 1) )
#     yaw =  np.deg2rad(0)
#
#     Rx = transf.rotation_matrix(roll, xaxis)
#     Ry = transf.rotation_matrix(pitch, yaxis)
#     Rz = transf.rotation_matrix(yaw, zaxis)
#     R = transf.concatenate_matrices(Rx, Ry, Rz)# order: Rz then Ry then Rx, e.g. np.dot(Rz[0:3,0:3],np.dot(Rx[0:3,0:3],Ry[0:3,0:3]))
#     # R is a relative rotation.
#
#     image_shape = in_sensor['image_shape']
#
#     """
#     in_sensor is of class xarray.core.dataset.Dataset.
#     Inside it, there are many xarray.DataArray s.
#     Like:
#     in_sensor.cam_x is an xarray.DataArray (in_sensor.cam_x.variable <xarray.Variable (npixels: 10000)> is array([1., ..., 1.]) )
#     in_sensor.image_shape is an xarray.DataArray
#     But, the in_sensor.cam_x  has Dimensions without coordinates: npixels. There are 10000 npixels, e.g. 10000 values.
#     You can not index a pixel in cam_x rather than just use the index of a pixel.
#     In the in_sensor.image_shape, the Coordinates  are regulat (image_dims) <U2 'nx' 'ny'.
#     Good reference to read about terminology is here https://docs.xarray.dev/en/stable/user-guide/terminology.html
#
#
#     in_sensor.coords
#     Coordinates:
#     * stokes_index  (stokes_index) <U1 'I' 'Q' 'U' 'V'
#     * image_dims    (image_dims) <U2 'nx' 'ny'
#
#     Variables:
#     wavelength
#     stokes
#     cam_x
#     cam_y
#     cam_z
#     cam_mu
#     cam_phi
#     image_shape
#     ray_mu
#     ray_phi
#     ray_x
#     ray_y
#     ray_z
#     pixel_index
#     ray_weight
#
#     """
#     out_sensor = in_sensor.copy(deep=True)
#
#
#
#     #load old parameteres:
#     old_lookat = in_sensor.attrs['lookat']
#     old_position = in_sensor.attrs['position']
#     old_direction = old_lookat - old_position
#     old_direction = norm(old_direction)
#
#     old_rotation_matrix = in_sensor.attrs['rotation_matrix'].reshape(3,3)
#     old_k = in_sensor.attrs['sensor_to_camera_transform_matrix'].reshape(3,3) # sensor_to_camera_transform_matrix
#
#     old_cam_dir_x =  np.dot(old_rotation_matrix,xaxis)
#     old_cam_dir_y =  np.dot(old_rotation_matrix,yaxis)
#     old_cam_dir_z =  np.dot(old_rotation_matrix,zaxis)
#     assert np.allclose(old_cam_dir_z,old_direction), "The vectors must be similar, chack the input."
#
#     # TODO, change out_sensor.attrs['position']
#     # TODO, change out_sensor.attrs['lookat']
#     # TODO, change out_sensor.attrs['rotation_matrix']
#     # TODO, change out_sensor.attrs['projection_matrix']
#     # TODO, change out_sensor.attrs['sensor_to_camera_transform_matrix']
#
#     """
#     ADD THE NOISE:
#
#     """
#     out_sensor.attrs['is_ideal_pointing'] = False
#     new_cam_dir_z =  np.dot(R[0:3,0:3],old_cam_dir_z)
#     report_angle_deviation = np.rad2deg( np.arccos( np.dot(new_cam_dir_z,old_cam_dir_z) ) )
#     print('The angle deviation form ideal pointing is {}[deg]'.format(report_angle_deviation))
#     # ----------------------------
#     #new_dir_z =  np.dot(R[0:3,0:3],zaxis)
#     #A_t = rad2deg*np.arccos(np.dot(zaxis,new_dir_z))
#     #assert A_t <= A , "Problem in pointing noise generation."
#     # ---------------------------------
#
#
#     old_pointing_vector = self.get_pointing_vector()  # for test porpuses
#     Ronly = self._T[0:3,0:3] # rotation
#     cam_dir_x =  np.dot(Ronly,xaxis)
#     cam_dir_y =  np.dot(Ronly,yaxis)
#     cam_dir_z =  np.dot(Ronly,zaxis)
#
#     Rx = transf.rotation_matrix(roll, cam_dir_x)
#     Ry = transf.rotation_matrix(pitch, cam_dir_y)
#     Rz = transf.rotation_matrix(yaw, cam_dir_z)
#     R_rel = transf.concatenate_matrices(Rx, Ry, Rz)# order: Ry then Rx
#
#     """Vadim implemented random noise in x,y directions independently.
#     Maybe the right nose model is different. Vadim should coordinate it with Alex.
#     some references:
#     1. https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
#     """
#     r = np.identity(4)
#     r[0:3,0:3] = self._T[0:3,0:3]
#     r_before_noise = r
#
#     r = np.dot(R_rel,r)
#
#     # only for test:
#     r_t = np.dot(R_rel.T,r)
#     test_dist = np.linalg.norm(r_before_noise  - r_t)
#     assert test_dist<1e-6 , "Problem with the noise apllied to the rotation matrix"
#
#     self._T[0:3,0:3] = r[0:3,0:3] # update the rotation with the noisy one, back to GT ratoadion do r = np.dot(self._rel_noisy.T,r)
#
#     NewUp = np.dot(R_rel[0:3,0:3],self._up)# update the up with the noisy one, back to GT (?)
#     # I had it befor but it is a bug, NewUp = np.dot(r[0:3,0:3],self._up)
#
#     NewOpticalDirection = np.dot(r[0:3,0:3],np.array(zaxis)) # cameras z axis.
#     # fined new lookAt point:
#     pinhole_point = self._T[0:3,3]
#
#     # intersection of a line with the ground surface (flat):
#     """p_co, p_no: define the plane:
#         p_co is a point on the plane (plane coordinate).
#         p_no is a normal vector defining the plane direction.
#         """
#     p_co = np.array([0,0,0])
#     p_no = np.array([0,0,1])
#     epsilon = 1e-6
#     u = NewOpticalDirection
#     Q = np.dot(p_no, u)
#
#
#
#     if abs(Q) > epsilon:
#         d = np.dot((p_co - pinhole_point),p_no)/Q
#         Newlookat = pinhole_point + (d*u)
#
#     else:
#         raise Exception("Can't find look at vector")
#
#
#     self._rel_noisy = R_rel
#     self._lookat = Newlookat
#     self._up = NewUp
#     self._was_noise_applied = True
#
#     # make another test:
#     new_pointing_vector = NewOpticalDirection
#     test_dist = np.linalg.norm(A_t  - rad2deg*np.arccos(np.dot(new_pointing_vector,old_pointing_vector)))
#     assert test_dist<1e-6 , "Problem accured in the when noise was applyed to pointing."
#     #print("Pointing error: {}[deg] error was simulated.".format(A_t))
#     ##R = np.dot(self._rel_noisy[0:3,0:3].T,self._T[0:3,0:3]) # rotation
#     ##test_dist = np.linalg.norm(r_before_noise  - R)
#     ##print(test_dist)
#
#
#     print(direction)


def show_results(sensor_dict):
    # see images:
    for instrument in sensor_dict:
        sensor_images = sensor_dict.get_images(instrument)

        PNCHANNELS = 1  # polarized channels
        pol_channels = ['I']
        if 'Q' in list(sensor_images[0].keys()) and 'U' in list(sensor_images[0].keys()):
            print(" The images are polarized")
            PNCHANNELS = 3
            pol_channels = ['I', 'Q', 'U']

        nrows = 2
        LN = len(sensor_images)
        if LN % nrows == 0:
            ncols = int(LN / nrows)
        else:
            if ((LN / nrows) > int(LN / nrows)):
                ncols = int(LN / nrows) + 1
            else:
                ncols = int(LN / nrows)

                # ------------------------------
        fontsize = 16
        for pol_channel in pol_channels:

            fig = plt.figure(figsize=(20, 10))
            fontsize = 16
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            max_ = 0
            min_ = 1e7

            for index, sensor in enumerate(sensor_images):
                img = sensor[pol_channel].T.data
                max_ = max(max_, img.max())
                min_ = min(min_, img.min())

            cmap = 'jet'

            for index, sensor in enumerate(sensor_images):
                img = sensor[pol_channel].T.data
                if pol_channel == 'I':
                    min_ = 0
                    cmap = 'gray'

                ii = index + 1
                ax = fig.add_subplot(nrows, ncols, ii)
                im = ax.imshow(img, cmap=cmap, vmin=min_, vmax=max_)
                title = "{}".format(index)
                ax.set_title(title, fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                plt.colorbar(im, cax=cax)
                ax.set_axis_off()

            fig.suptitle("{}: channel {}".format(instrument, pol_channel), size=16, y=0.95)

    plt.show()

    print('done')


def convertStocks(sensor_dict, r_sat, GSD, method='meridian2camera'):
    """
    TODO
    returns sensor_dict_out = copy.deepcopy(sensor_dict)

    """

    sensor_dict_out = copy.deepcopy(sensor_dict)

    VISUALIZE = False
    assert len(sensor_dict) == 1, "Currently doesn't soppurt more than 1 instrument"
    FIRST_SENSOR = sensor_dict.get_image(list(sensor_dict.keys())[0], 0)
    cnx, cny = FIRST_SENSOR.dims['imgdim0'], FIRST_SENSOR.dims['imgdim1']  # TODO what if the camera multiband?
    npix = cnx * cny
    nviews = len(sensor_dict.get_images(list(sensor_dict.keys())[0]))
    sensor_list = sensor_dict[list(sensor_dict.keys())[0]]['sensor_list']
    # list of <xarray.Dataset>
    one_sensor_example = sensor_list[0]

    stokes_converted = np.zeros([nviews, 3, cnx, cny])
    # stokes_converted size is [view,stokes,nx,ny]

    for instrument in sensor_dict:
        sensor_images = sensor_dict.get_images(instrument)
        sensor_list = sensor_dict[instrument]['sensor_list']

        PNCHANNELS = 1  # polarized channels
        pol_channels = ['I']
        if 'Q' in list(sensor_images[0].keys()) and 'U' in list(sensor_images[0].keys()):
            print(" The images are polarized")
            PNCHANNELS = 3
            pol_channels = ['I', 'Q', 'U']
        else:
            raise Exception("Stokes convertion can't work on unpolarized images.")

        for sensor_index, sensor_image in enumerate(sensor_images):
            stokes = np.dstack([sensor_image[pol_channel].data for pol_channel in pol_channels])
            sensor = sensor_list[sensor_index]

            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------

            # get theta filename:
            current_lookat = sensor.lookat
            current_origin = sensor.position
            current_fov = float_round(one_sensor_example.fov_deg)
            current_gsd = GSD

            lookat_str = '_'.join(str(int(np.round(x * 1000))) for x in current_lookat)
            origin_str = '_'.join(str(np.round(x, 3)) for x in current_origin)
            dir_path = os.path.join('..', 'polar_convert_angles')
            filename = os.path.join(dir_path,
                                    'theta_pixx_' + str(cnx) + '_pixy_' + str(cny) + '_gsd_' + str(
                                        current_gsd) + '_lookat_' + lookat_str + '_origin_' + origin_str + '.pkl')

            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------

            # get ray directions:

            ray_mus = sensor.ray_mu
            ray_phis = sensor.ray_phi
            ray_xs = sensor.ray_x
            ray_ys = sensor.ray_y
            ray_zs = sensor.ray_z

            # angles to rays convertion
            PHI = ray_phis
            RAY_Z = ray_mus
            RAY_X = np.sin(np.arccos(ray_mus)) * np.cos(ray_phis)
            RAY_Y = np.sin(np.arccos(ray_mus)) * np.sin(ray_phis)

            # scale_factor=2*scale
            if (VISUALIZE):
                inte_length = 200
                mlab.quiver3d(ray_xs[100], ray_ys[100], ray_zs[100], inte_length * RAY_X[100],
                              inte_length * RAY_Y[100], inte_length * RAY_Z[100], \
                              line_width=0.001, color=(0.6, 1.0, 0), opacity=0.2,
                              scale_factor=1)  # ,scale_factor=2*scale

                # at3d.sensor.show_sensors(sensor_list, scale = 50, axisWidth = 1.0, axisLenght=30, Show_Rays =  True, FullCone = True)
                mlab.orientation_axes()

            # other view parameteres:
            lookat = sensor.attrs['lookat']
            position = sensor.attrs['position']
            optical_axis_direction = lookat - position
            # calc the principle ray direction
            prd = optical_axis_direction / np.linalg.norm(optical_axis_direction)

            rotation_matrix = sensor.attrs['rotation_matrix'].reshape(3, 3)

            cam_dir_x = np.dot(rotation_matrix, xaxis)
            cam_dir_y = np.dot(rotation_matrix, yaxis)
            cam_dir_z = np.dot(rotation_matrix, zaxis)

            left_of_polarizer = cam_dir_y.copy()  #
            left_of_polarizer = left_of_polarizer / np.linalg.norm(left_of_polarizer)

            # calc polaizer direction at the principle ray direction (prd)
            polarizer_dir_at_prd = np.cross(left_of_polarizer,
                                            prd)  # this makes a perfect alinment of the polarizer with cameras x-axis

            # rel_ang is the angle of rotation that the scocks vector in meridian frame should be
            # transformed to get the stocks vector in the camera frame.

            zenith_dir = np.array([0, 0, 1])

            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    theta_rad_mat = pickle.load(f)
                    print("Theta matrix file of {} was read.".format(sensor_index))
            else:
                print('Converting {}'.format(sensor_index))
                """
                that how I want to setup the polarizer, to aligned 
                with camera xaxis at principle point direction.
                It is where polarizer_angle = 0 # ged
                """
                # the inverse of flatten - I.flatten().reshape(I.shape, order='C')
                theta_rad_mat = np.zeros([cnx, cny])
                for index, (d_x, d_y, d_z, origin_x, origin_y, origin_z, phi_dir) in enumerate(zip(RAY_X.data,
                                                                                                   RAY_Y.data,
                                                                                                   RAY_Z.data,
                                                                                                   ray_xs.data,
                                                                                                   ray_ys.data,
                                                                                                   ray_zs.data,
                                                                                                   PHI.data)):
                    scale_for_lookat = 0.1 * origin_z + 1  # km
                    ray_origin = np.array([origin_x, origin_y, origin_z])
                    ray_dir = np.array([d_x, d_y, d_z])

                    lookat_pix = ray_origin + scale_for_lookat * ray_dir  # we have a freadom to play with it i think.
                    # pick 3 point on the plane:
                    p1 = lookat_pix
                    p2 = ray_origin
                    p3 = lookat_pix + zenith_dir * scale_for_lookat

                    # These two vectors are in the plane
                    v1 = p3 - p1
                    v2 = p2 - p1
                    # the cross product is a vector normal to the plane
                    r = np.cross(v1, v2)
                    if (np.all(r == 0)):
                        # V1 parallel to V2 -> meridian aligned with camera reference.
                        theta_rad = 0
                        # update polarizer_dir as the polarizer is spherical and any ray is perpendicular to its surface
                        polarizer_dir = np.cross(left_of_polarizer, ray_dir)
                        polarizer_phi = (np.arctan2(polarizer_dir[1], polarizer_dir[0]) + np.pi).astype(np.float64)

                    else:
                        # continue to meridian coordinates:
                        z = ray_dir
                        r = r / np.linalg.norm(r)
                        l = np.cross(z, r)
                        assert np.allclose(z, np.cross(r, l)), "Bad calculation of vectors directions."
                        # find theta
                        # update polarizer_dir as the polarizer is spherical and any ray is perpendicular to its surface
                        polarizer_dir = np.cross(left_of_polarizer, ray_dir)
                        polarizer_phi = (np.arctan2(polarizer_dir[1], polarizer_dir[0]) + np.pi).astype(np.float64)

                        cos = np.dot(polarizer_dir, l)
                        cos = np.clip(cos, -1, 1)
                        # I had a problem here, the cos can be something like -1.000000000000002 and this is not valid number for acos
                        theta_rad = np.arccos(cos)  # polarizer_dir for 0[deg] is lo direction.

                    """
                            See page 51 in the book Scattering, Absorption, and Emission of Light by Small Particles.
                            https://onlinelibrary.wiley.com/doi/pdf/10.1002/9783527618156.

                            The rotation angle is theta_rad.

                    """
                    # Find dphi -  to dicide if the angle is theta_rad or -theta_rad.
                    d = phi_dir - polarizer_phi
                    if (d > 2 * np.pi):
                        d = d - 2 * np.pi

                    if ((d <= np.pi) and (d >= 0)):
                        dphi = 1.0

                    elif ((d <= -np.pi) and (d >= -2 * np.pi)):
                        dphi = 1.0

                    else:
                        dphi = -1.0

                    theta_rad *= dphi  # it is very important to give here the sign of theta.

                    xpix_ind, ypix_ind = np.unravel_index(index, (cnx, cny), order='F')

                    theta_rad_mat[xpix_ind, ypix_ind] = theta_rad

                # save theta_rad_mat for future runs
                if not os.path.exists(dir_path):
                    # Create a new directory because it does not exist
                    os.makedirs(dir_path)
                    print("The directory 'polar_convert_angles' was created.")
                with open(filename, 'wb') as outfile:
                    pickle.dump(theta_rad_mat, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Theta matrix file was saved.")

            # out of the "if matrix exists"
            for index, (d_x, d_y, d_z, origin_x, origin_y, origin_z, phi_dir, theta_rad) in enumerate(zip(RAY_X.data,
                                                                                                          RAY_Y.data,
                                                                                                          RAY_Z.data,
                                                                                                          ray_xs.data,
                                                                                                          ray_ys.data,
                                                                                                          ray_zs.data,
                                                                                                          PHI.data,
                                                                                                          theta_rad_mat.flatten())):
                R_theta = np.array([[1, 0, 0, 0], [0, np.cos(2 * theta_rad), np.sin(2 * theta_rad), 0],
                                    [0, -np.sin(2 * theta_rad), np.cos(2 * theta_rad), 0], [0, 0, 0, 1]])

                # find 2d indexes:
                xpix_ind, ypix_ind = np.unravel_index(index, (cnx, cny))
                # S # this pixel stokes vector:
                stokes_I = stokes[xpix_ind, ypix_ind, 0]
                stokes_Q = stokes[xpix_ind, ypix_ind, 1]
                stokes_U = stokes[xpix_ind, ypix_ind, 2]

                S = np.vstack([stokes_I,
                               stokes_Q,
                               stokes_U,
                               0])
                # stocks vector

                S_at_pixel = S
                # now S has 3 elements.

                if (method == 'meridian2camera'):

                    Sconvertaed_at_pixel = np.dot(R_theta,
                                                  S_at_pixel)  # transfer the reference frame from meridian plane to camera plane

                elif (method == 'camera2meridian'):
                    Sconvertaed_at_pixel = np.dot(np.linalg.inv(R_theta),
                                                  S_at_pixel)  # inverse transfer the reference frame from meridian plane to camera plane

                else:
                    raise Exception("Unknown method of stokes convertion.")

                Sconvertaed_at_pixel = Sconvertaed_at_pixel[:-1]  # get rid of V=0

                if (any(np.isnan(elem) for elem in Sconvertaed_at_pixel)):
                    print('-------------------------------------')
                    print(ray_dir)
                    print(theta_rad)
                    print(R_theta)
                    print('.....')
                    print(Sconvertaed_at_pixel)
                    raise Exception("Bad rotation of the meridian/camera plane")

                # update the big matrix stock_converted:
                # stokes_converted size is [view,stokes,nx,ny]
                stokes_converted[sensor_index, :, xpix_ind, ypix_ind] = \
                    np.squeeze(Sconvertaed_at_pixel)

            # Here we want to update the images of the sensor_dict, spesificaly: I, Q, U.
            # here we still in the sensor_inxed loop.

            sensor_dict_out[instrument]['sensor_list'][sensor_index]['Q'].data = stokes_converted[
                sensor_index, 1, ...].flatten(order='F')
            sensor_dict_out[instrument]['sensor_list'][sensor_index]['U'].data = stokes_converted[
                sensor_index, 2, ...].flatten(order='F')

    return sensor_dict_out


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

def generate_random_sun_angles_for_lat(Lat):
    day_num = np.random.default_rng().integers(1, high=365, endpoint=True)
    LST = np.random.default_rng().integers(0, high=23, endpoint=True)

    delta = 23.45 * np.sin(np.deg2rad((360 / 365) * (284 + day_num)))
    h = (LST - 12) * 15
    sun_alt = np.rad2deg(
        np.arcsin(np.sin(np.deg2rad(Lat)) * np.sin(np.deg2rad(delta)) +
                  np.cos(np.deg2rad(Lat)) * np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(h))))
    sun_azimuth = np.rad2deg(np.arcsin(np.cos(np.deg2rad(delta)) * np.sin(np.deg2rad(h)) / np.cos(np.deg2rad(sun_alt))))
    while (not np.isreal(sun_alt)) or (not np.isreal(sun_azimuth)) or (sun_alt < 0):
        day_num = np.random.default_rng().integers(1, high=365, endpoint=True)
        LST = np.random.default_rng().integers(0, high=23, endpoint=True)

        delta = 23.45 * np.sin(np.deg2rad((360 / 365) * (284 + day_num)))
        h = (LST - 12) * 15
        sun_alt = np.rad2deg(
            np.arcsin(np.sin(np.deg2rad(Lat)) * np.sin(np.deg2rad(delta)) +
                      np.cos(np.deg2rad(Lat)) * np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(h))))
        sun_azimuth = np.rad2deg(
            np.arcsin(np.cos(np.deg2rad(delta)) * np.sin(np.deg2rad(h)) / np.cos(np.deg2rad(sun_alt))))
    sun_zenith = sun_alt + 90
    return sun_azimuth, sun_zenith


def calc_image_in_scattering_plane(sensor, sensor_name, sun_azimuth, sun_zenith, theta_dir, path_stamp):
    theta_filename = os.path.join(theta_dir, path_stamp,
                                  sensor_name + '_sa' + str(sun_azimuth) + '_sz' + str(sun_zenith) + '.pkl')
    if os.path.exists(theta_filename):
        with open(theta_filename, 'rb') as f:
            theta_rad_mat = pickle.load(f)
            print("Theta matrix file of {} was read for projection {}.".format(sensor_name, path_stamp))
    else:
        print('Converting {} for projection {}'.format(sensor_name, path_stamp))
        zenith_dir = np.array([0, 0, 1])
        PHI = sensor.ray_phi.data
        THETA = np.arccos(sensor.ray_mu.data)  # mu is defined as -z !!!
        resolution = sensor.image_dims.data
        PHI = PHI.reshape(resolution, order='F')
        THETA = THETA.reshape(resolution, order='F')
        MU = np.cos(THETA)
        RAY_Z = -MU
        RAY_X = np.sin(np.arccos(MU)) * np.cos(PHI)
        RAY_Y = np.sin(np.arccos(MU)) * np.sin(PHI)

        theta_rad_mat = np.zeros_like(RAY_X)

        alpha = (180 - sun_zenith) * np.pi / 180
        beta = sun_azimuth * np.pi / 180
        sun_dir = np.array([np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta), np.cos(alpha)])

        for index, (d_x, d_y, d_z, phi_dir) in enumerate(zip(RAY_X.flatten(),
                                                             RAY_Y.flatten(),
                                                             RAY_Z.flatten(),
                                                             PHI.flatten())):
            ray_dir = np.array([d_x, d_y, d_z])

            persp_cam_vec = np.cross(zenith_dir, ray_dir)
            persp_cam_vec = persp_cam_vec / np.linalg.norm(persp_cam_vec)
            persp_cam_phi = (np.arctan2(persp_cam_vec[1], persp_cam_vec[0]) + np.pi).astype(
                np.float64)  # phi_dir

            scat_vec = np.cross(sun_dir, ray_dir)
            scat_vec = scat_vec / np.linalg.norm(scat_vec)
            scat_phi = (np.arctan2(scat_vec[1], scat_vec[0]) + np.pi).astype(np.float64)

            cos = np.dot(persp_cam_vec, scat_vec)
            theta_rad = np.arccos(cos)

            d = persp_cam_phi - scat_phi
            if (d > 2 * np.pi):
                d = d - 2 * np.pi

            if ((d <= np.pi) and (d >= 0)):
                dphi = 1.0

            elif ((d <= -np.pi) and (d >= -2 * np.pi)):
                dphi = 1.0

            else:
                dphi = -1.0

            theta_rad *= dphi

            xpix_ind, ypix_ind = np.unravel_index(index, (resolution[0], resolution[1]))
            theta_rad_mat[xpix_ind, ypix_ind] = theta_rad
        # save theta_rad_mat for future runs
        if not os.path.exists(theta_dir):
            # Create a new directory because it does not exist
            os.makedirs(theta_dir)
            print("The directory for saving theta matrices was created.")
        if not os.path.exists(os.path.join(theta_dir, path_stamp)):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(theta_dir, path_stamp))
            print("The directory for saving theta matrices for projection {} was created.".format(path_stamp))
        with open(filename, 'wb') as outfile:
            pickle.dump(theta_rad_mat, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            print("Theta matrix file was saved.")

            theta_filename = os.path.join(theta_dir, path_stamp, sensor_name + '.pkl')
            if os.path.exists(theta_filename):
                with open(theta_filename, 'rb') as f:
                    theta_rad_mat = pickle.load(f)
                    print("Theta matrix file of {} was read for projection {}.".format(sensor_name, path_stamp))

    image = images[proj_ind]
    scatter_image = np.zeros_like(image)
    stokes_V = 0
    for index, (stokes_I, stokes_Q, stokes_U) in enumerate(zip(image[0].flatten(),
                                                               image[1].flatten(),
                                                               image[2].flatten())):
        theta_rad = theta_rad_mat.flatten()[index]
        R_theta = np.array([[1, 0, 0, 0], [0, np.cos(2 * theta_rad), np.sin(2 * theta_rad), 0],
                            [0, -np.sin(2 * theta_rad), np.cos(2 * theta_rad), 0], [0, 0, 0, 1]])

        xpix_ind, ypix_ind = np.unravel_index(index, (resolution[0], resolution[1]))
        # S # this pixel stokes vector:

        S = np.vstack([stokes_I,
                       stokes_Q,
                       stokes_U,
                       stokes_V])  # stokes vector

        S_at_pixel = S
        # now S has 4 elements.

        Sconvertaed_at_pixel = np.dot(R_theta, S_at_pixel)
        S_3 = Sconvertaed_at_pixel[0:3]
        scatter_image[:, xpix_ind, ypix_ind] = np.squeeze(S_3)
    return scatter_image


def main():
    file_name = 'Radiances_to_calibrate_stokes_conversions_1.nc'
    sensor_dict, solver_dict, rte_grid = at3d.util.load_forward_model(file_name)
    method = 'meridian2camera'
    fontsize = 16

    r_sat = 500  # km
    GSD = 0.02
    sensor_dict_camera = convertStocks(sensor_dict, r_sat, GSD, method)
    # stokes_converted_back = convertStocks(sensor_dict, r_sat, method)

    # see images:
    for instrument in sensor_dict:
        meridian_sensor_images = sensor_dict.get_images(instrument)
        camera_sensor_images = sensor_dict_camera.get_images(instrument)

        sensor_list = sensor_dict[instrument]['sensor_list']

        PNCHANNELS = 1  # polarized channels
        pol_channels = ['I']
        if 'Q' in list(meridian_sensor_images[0].keys()) and 'U' in list(meridian_sensor_images[0].keys()):
            print(" The images are polarized")
            PNCHANNELS = 3
            pol_channels = ['I', 'Q', 'U']

        nrows = 2
        LN = len(meridian_sensor_images)
        if LN % nrows == 0:
            ncols = int(LN / nrows)
        else:
            if ((LN / nrows) > int(LN / nrows)):
                ncols = int(LN / nrows) + 1
            else:
                ncols = int(LN / nrows)

                # ------------------------------
        fontsize = 16

        for sensor_index, (meridian_sensor_image, camera_sensor_image) in enumerate(
                zip(meridian_sensor_images, camera_sensor_images)):
            stokes_maridian = np.dstack([meridian_sensor_image[pol_channel].data for pol_channel in pol_channels])
            stokes_camera = np.dstack([camera_sensor_image[pol_channel].data for pol_channel in pol_channels])

            fig = plt.figure(figsize=(20, 10))
            ii = 1
            for pol_index, pol_channel in enumerate(pol_channels):
                ax = fig.add_subplot(4, 3, ii)

                img_meridian = stokes_maridian[..., pol_index]
                img_camera = stokes_camera[..., pol_index]

                min_ = min(img_meridian.min(), img_camera.min())
                max_ = max(img_meridian.max(), img_camera.max())

                im = ax.imshow(img_meridian, cmap='jet', vmin=min_, vmax=max_)
                title = "meridian {} {}".format(sensor_index, pol_channel)
                ax.set_title(title, fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                plt.colorbar(im, cax=cax)
                ax.set_axis_off()

                # -------camera----------------------------
                ax = fig.add_subplot(4, 3, ii + 3)

                im = ax.imshow(img_camera, cmap='jet', vmin=min_, vmax=max_)
                title = "camera {} {}".format(sensor_index, pol_channel)
                ax.set_title(title, fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                plt.colorbar(im, cax=cax)
                ax.set_axis_off()

                # -------dolp----------------------------
                ax = fig.add_subplot(4, 3, ii + 3)

                im = ax.imshow(img_camera, cmap='jet', vmin=min_, vmax=max_)
                title = "camera {} {}".format(sensor_index, pol_channel)
                ax.set_title(title, fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                plt.colorbar(im, cax=cax)
                ax.set_axis_off()

                ii = ii + 1
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------

    plt.show()
    print('done')


if __name__ == '__main__':
    main()