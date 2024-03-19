import cv2
import at3d
import numpy as np
import xarray as xr
from collections import OrderedDict
import pylab as py
import matplotlib.pyplot as plt
import pickle
import os
import scipy.io as sio
import netCDF4
import re
import csv
import glob
from scipy import ndimage
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
from multiprocessing import Pool
from itertools import repeat
from CloudCTUtils import *
from CloudCT_NoiseUtils import *
import matplotlib

matplotlib.use('TkAgg')

# constants
r_earth = 6371.0  # km
origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]


def main():
    AirMSPI_Test(test_ind=5, surface_wind_speed=10., render_cloud=True, zeta=None)
    # AirMSPI_Diner_test()
    # AirMSPI_wind_test()
    # AirMSPI_albedo_test()
    print('finished successfully.')


def AirMSPI_Diner_test():
    mass_error = lambda ext_est, ext_gt, eps=1e-6: (np.linalg.norm(ext_gt, ord=1) - np.linalg.norm(ext_est, ord=1)) / (np.linalg.norm(ext_gt, ord=1) + eps)
    test_ind = 1
    wind_vec = np.array([10.])
    zeta_vec = np.array([0.1, 0.4, 0.8, 1])
    pixel_precent = 0.11
    gt_I_list = []
    I_list = []
    gt_DoLP_list = []
    DoLP_list = []
    gt_AoLP_scat_list = []
    AoLP_scat_list = []
    I_delta_list = []
    DoLP_delta_list = []
    AoLP_delta_list = []
    title_list = []
    for wind in wind_vec:
        for zeta in zeta_vec:
            title_list.append(f'wind={wind}[m/s]\n zeta={zeta}')
            gt_I, I, gt_DoLP, DoLP, gt_AoLP_scat, AoLP_scat = AirMSPI_Test(test_ind, wind, render_cloud=False, zeta=zeta)
            if wind == wind_vec[0] and zeta == zeta_vec[0]:
                rand_ind = np.random.choice(np.arange(len(gt_I[0,:,:].ravel())), size=int(pixel_precent * len(gt_I[0,:,:].ravel())),
                                            replace=False)
            gt_I_list.append(gt_I.reshape(gt_I.shape[:-2]+tuple([-1]))[:, rand_ind])
            I_list.append(I.reshape(I.shape[:-2]+tuple([-1]))[:, rand_ind])
            gt_DoLP_list.append(gt_DoLP.reshape(gt_DoLP.shape[:-2]+tuple([-1]))[:, rand_ind])
            DoLP_list.append(DoLP.reshape(DoLP.shape[:-2]+tuple([-1]))[:, rand_ind])
            gt_AoLP_scat_list.append(gt_AoLP_scat.reshape(gt_AoLP_scat.shape[:-2]+tuple([-1]))[:, rand_ind])
            AoLP_scat_list.append(AoLP_scat.reshape(AoLP_scat.shape[:-2]+tuple([-1]))[:, rand_ind])
            delta_I = [mass_error(im, gt_im) for im, gt_im in zip(I_list[-1], gt_I_list[-1])]
            delta_D = [mass_error(im, gt_im) for im, gt_im in zip(DoLP_list[-1], gt_DoLP_list[-1])]
            delta_A = [mass_error(im, gt_im) for im, gt_im in zip(AoLP_scat_list[-1], gt_AoLP_scat_list[-1])]
            I_delta_list.append(np.array(delta_I))
            DoLP_delta_list.append(np.array(delta_D))
            AoLP_delta_list.append(np.array(delta_A))

    gt_I_per_im = np.array(gt_I_list).transpose((1, 0, 2))
    I_per_im = np.array(I_list).transpose((1, 0, 2))
    gt_DoLP_per_im = np.array(gt_DoLP_list).transpose((1, 0, 2))
    DoLP_per_im = np.array(DoLP_list).transpose((1, 0, 2))
    gt_AoLP_per_im = np.array(gt_AoLP_scat_list).transpose((1, 0, 2))
    AoLP_per_im = np.array(AoLP_scat_list).transpose((1, 0, 2))
    I_delta_per_im = np.array(I_delta_list).transpose((1, 0))
    DoLP_delta_per_im = np.array(DoLP_delta_list).transpose((1, 0))
    AoLP_delta_per_im = np.array(AoLP_delta_list).transpose((1, 0))
    for im_ind, (curr_gt_I, curr_I, curr_gt_D, curr_D, curr_gt_A, curr_A, curr_delta_I, curr_delta_D, curr_delta_A) in \
        enumerate(zip(gt_I_per_im, I_per_im, gt_DoLP_per_im, DoLP_per_im, gt_AoLP_per_im, AoLP_per_im,
                      I_delta_per_im, DoLP_delta_per_im, AoLP_delta_per_im)):

        fig, axarr = plt.subplots(3, len(wind_vec)*len(zeta_vec), figsize=(20, 20))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        axarr = axarr.flatten()
        for ax, title, gt_param, est_param in zip(axarr[:(len(wind_vec)*len(zeta_vec))], title_list, curr_gt_I, curr_I):
            max_val = 0.032 #max(gt_param.max(), est_param.max())
            min_val = 0 #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            ax.set_title(f'I \n' + title)
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        for ax, title, gt_param, est_param in zip(axarr[(len(wind_vec)*len(zeta_vec)):2*(len(wind_vec)*len(zeta_vec))], title_list, curr_gt_D, curr_D):
            max_val = 1. #max(gt_param.max(), est_param.max())
            min_val = 0. #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            ax.set_title(f'DoLP \n' + title)
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        for ax, title, gt_param, est_param in zip(axarr[2*(len(wind_vec)*len(zeta_vec)):], title_list, curr_gt_A, curr_A):
            max_val = 100 #max(gt_param.max(), est_param.max())
            min_val = 80 #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            ax.set_title(f'AoLP scatter \n' + title)
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        fig.suptitle('image #'+str(im_ind)+' scatter plots for different wind speeds and zetas', size=16, y=0.95)

    plt.show()
    a=5


def AirMSPI_wind_test():
    mass_error = lambda ext_est, ext_gt, eps=1e-6: (np.linalg.norm(ext_gt, ord=1) - np.linalg.norm(ext_est, ord=1)) / (np.linalg.norm(ext_gt, ord=1) + eps)
    test_ind = 1
    wind_vec = np.array([1, 3, 6, 9, 12, 14])
    pixel_precent = 0.11
    gt_I_list = []
    I_list = []
    gt_DoLP_list = []
    DoLP_list = []
    gt_AoLP_scat_list = []
    AoLP_scat_list = []
    I_delta_list = []
    DoLP_delta_list = []
    AoLP_delta_list = []
    for wind in wind_vec:
        gt_I, I, gt_DoLP, DoLP, gt_AoLP_scat, AoLP_scat = AirMSPI_Test(test_ind, wind, render_cloud=False)
        if wind == wind_vec[0]:
            rand_ind = np.random.choice(np.arange(len(gt_I[0,:,:].ravel())), size=int(pixel_precent * len(gt_I[0,:,:].ravel())),
                                        replace=False)
        gt_I_list.append(gt_I.reshape(gt_I.shape[:-2]+tuple([-1]))[:, rand_ind])
        I_list.append(I.reshape(I.shape[:-2]+tuple([-1]))[:, rand_ind])
        gt_DoLP_list.append(gt_DoLP.reshape(gt_DoLP.shape[:-2]+tuple([-1]))[:, rand_ind])
        DoLP_list.append(DoLP.reshape(DoLP.shape[:-2]+tuple([-1]))[:, rand_ind])
        gt_AoLP_scat_list.append(gt_AoLP_scat.reshape(gt_AoLP_scat.shape[:-2]+tuple([-1]))[:, rand_ind])
        AoLP_scat_list.append(AoLP_scat.reshape(AoLP_scat.shape[:-2]+tuple([-1]))[:, rand_ind])
        delta_I = [mass_error(im, gt_im) for im, gt_im in zip(I_list[-1], gt_I_list[-1])]
        delta_D = [mass_error(im, gt_im) for im, gt_im in zip(DoLP_list[-1], gt_DoLP_list[-1])]
        delta_A = [mass_error(im, gt_im) for im, gt_im in zip(AoLP_scat_list[-1], gt_AoLP_scat_list[-1])]
        I_delta_list.append(np.array(delta_I))
        DoLP_delta_list.append(np.array(delta_D))
        AoLP_delta_list.append(np.array(delta_A))

    gt_I_per_im = np.array(gt_I_list).transpose((1, 0, 2))
    I_per_im = np.array(I_list).transpose((1, 0, 2))
    gt_DoLP_per_im = np.array(gt_DoLP_list).transpose((1, 0, 2))
    DoLP_per_im = np.array(DoLP_list).transpose((1, 0, 2))
    gt_AoLP_per_im = np.array(gt_AoLP_scat_list).transpose((1, 0, 2))
    AoLP_per_im = np.array(AoLP_scat_list).transpose((1, 0, 2))
    I_delta_per_im = np.array(I_delta_list).transpose((1, 0))
    DoLP_delta_per_im = np.array(DoLP_delta_list).transpose((1, 0))
    AoLP_delta_per_im = np.array(AoLP_delta_list).transpose((1, 0))
    for im_ind, (curr_gt_I, curr_I, curr_gt_D, curr_D, curr_gt_A, curr_A, curr_delta_I, curr_delta_D, curr_delta_A) in \
        enumerate(zip(gt_I_per_im, I_per_im, gt_DoLP_per_im, DoLP_per_im, gt_AoLP_per_im, AoLP_per_im,
                      I_delta_per_im, DoLP_delta_per_im, AoLP_delta_per_im)):

        fig, axarr = plt.subplots(3, len(wind_vec), figsize=(20, 20))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        axarr = axarr.flatten()
        for ax, gt_param, est_param, wind in zip(axarr[:len(wind_vec)], curr_gt_I, curr_I, wind_vec):
            max_val = 0.032 #max(gt_param.max(), est_param.max())
            min_val = 0 #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            ax.set_title(f'I \n speed=' + str(wind) + '[m/s]')
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        for ax, gt_param, est_param, wind in zip(axarr[len(wind_vec):2*len(wind_vec)], curr_gt_D, curr_D, wind_vec):
            max_val = 0.85 #max(gt_param.max(), est_param.max())
            min_val = 0.15 #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            ax.set_title(f'DoLP \n speed=' + str(wind) + '[m/s]')
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        for ax, gt_param, est_param, wind in zip(axarr[2*len(wind_vec):], curr_gt_A, curr_A, wind_vec):
            max_val = 100 #max(gt_param.max(), est_param.max())
            min_val = 80 #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            ax.set_title(f'AoLP \n speed=' + str(wind) + '[m/s]')
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        fig.suptitle('image #'+str(im_ind)+' scatter plots for different wind speeds', size=16, y=0.95)

        # fig, axarr = plt.subplots(1, 3, figsize=(20, 20))
        # fig.subplots_adjust(hspace=0.2, wspace=0.2)
        # axarr = axarr.flatten()
        # axarr[0].plot(wind_vec, curr_delta_I)
        # axarr[0].set_title('I mass error values as a function of wind speed')
        # axarr[0].set_ylabel('mass error value', fontsize=14)
        # axarr[0].set_xlabel('wind speed [m/s]', fontsize=14)
        # axarr[1].plot(wind_vec, curr_delta_D)
        # axarr[1].set_title('DoLP mass error values as a function of wind speed')
        # axarr[1].set_ylabel('mass error value', fontsize=14)
        # axarr[1].set_xlabel('wind speed [m/s]', fontsize=14)
        # axarr[2].plot(wind_vec, curr_delta_A)
        # axarr[2].set_title('AoLP mass error values as a function of wind speed')
        # axarr[2].set_ylabel('mass error value', fontsize=14)
        # axarr[2].set_xlabel('wind speed [m/s]', fontsize=14)
        # fig.suptitle('image #' + str(im_ind) + ' mass error values', size=16, y=0.95)
    plt.show()


def AirMSPI_albedo_test():
    mass_error = lambda ext_est, ext_gt, eps=1e-6: (np.linalg.norm(ext_gt, ord=1) - np.linalg.norm(ext_est, ord=1)) / (np.linalg.norm(ext_gt, ord=1) + eps)
    test_ind = 1
    wind_vec = np.array([0.01, 0.02, 0.03])  #np.array([0.02, 0.03, 0.1, 0.2, 0.5])
    pixel_precent = 0.11
    gt_I_list = []
    I_list = []
    gt_DoLP_list = []
    DoLP_list = []
    gt_AoLP_scat_list = []
    AoLP_scat_list = []
    I_delta_list = []
    DoLP_delta_list = []
    AoLP_delta_list = []
    for wind in wind_vec:
        gt_I, I, gt_DoLP, DoLP, gt_AoLP_scat, AoLP_scat = AirMSPI_Test(test_ind, wind, render_cloud=False)
        if wind == wind_vec[0]:
            rand_ind = np.random.choice(np.arange(len(gt_I[0,:,:].ravel())), size=int(pixel_precent * len(gt_I[0,:,:].ravel())),
                                        replace=False)
        gt_I_list.append(gt_I.reshape(gt_I.shape[:-2]+tuple([-1]))[:, rand_ind])
        I_list.append(I.reshape(I.shape[:-2]+tuple([-1]))[:, rand_ind])
        gt_DoLP_list.append(gt_DoLP.reshape(gt_DoLP.shape[:-2]+tuple([-1]))[:, rand_ind])
        DoLP_list.append(DoLP.reshape(DoLP.shape[:-2]+tuple([-1]))[:, rand_ind])
        gt_AoLP_scat_list.append(gt_AoLP_scat.reshape(gt_AoLP_scat.shape[:-2]+tuple([-1]))[:, rand_ind])
        AoLP_scat_list.append(AoLP_scat.reshape(AoLP_scat.shape[:-2]+tuple([-1]))[:, rand_ind])
        delta_I = [mass_error(im, gt_im) for im, gt_im in zip(I_list[-1], gt_I_list[-1])]
        delta_D = [mass_error(im, gt_im) for im, gt_im in zip(DoLP_list[-1], gt_DoLP_list[-1])]
        delta_A = [mass_error(im, gt_im) for im, gt_im in zip(AoLP_scat_list[-1], gt_AoLP_scat_list[-1])]
        I_delta_list.append(np.array(delta_I))
        DoLP_delta_list.append(np.array(delta_D))
        AoLP_delta_list.append(np.array(delta_A))

    gt_I_per_im = np.array(gt_I_list).transpose((1, 0, 2))
    I_per_im = np.array(I_list).transpose((1, 0, 2))
    gt_DoLP_per_im = np.array(gt_DoLP_list).transpose((1, 0, 2))
    DoLP_per_im = np.array(DoLP_list).transpose((1, 0, 2))
    gt_AoLP_per_im = np.array(gt_AoLP_scat_list).transpose((1, 0, 2))
    AoLP_per_im = np.array(AoLP_scat_list).transpose((1, 0, 2))
    I_delta_per_im = np.array(I_delta_list).transpose((1, 0))
    DoLP_delta_per_im = np.array(DoLP_delta_list).transpose((1, 0))
    AoLP_delta_per_im = np.array(AoLP_delta_list).transpose((1, 0))
    for im_ind, (curr_gt_I, curr_I, curr_gt_D, curr_D, curr_gt_A, curr_A, curr_delta_I, curr_delta_D, curr_delta_A) in \
        enumerate(zip(gt_I_per_im, I_per_im, gt_DoLP_per_im, DoLP_per_im, gt_AoLP_per_im, AoLP_per_im,
                      I_delta_per_im, DoLP_delta_per_im, AoLP_delta_per_im)):

        fig, axarr = plt.subplots(3, len(wind_vec), figsize=(20, 20))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        axarr = axarr.flatten()
        for ax, gt_param, est_param, wind in zip(axarr[:len(wind_vec)], curr_gt_I, curr_I, wind_vec):
            max_val = 0.1 #max(gt_param.max(), est_param.max())
            min_val = 0 #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            #ax.set_title(f'I \n speed=' + str(wind) + '[m/s]')
            ax.set_title(f'I \n albedo=' + str(wind))
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        for ax, gt_param, est_param, wind in zip(axarr[len(wind_vec):2*len(wind_vec)], curr_gt_D, curr_D, wind_vec):
            max_val = 0.85 #max(gt_param.max(), est_param.max())
            min_val = 0.15 #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            # ax.set_title(f'DoLP \n speed=' + str(wind) + '[m/s]')
            ax.set_title(f'DoLP \n albedo=' + str(wind))
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        for ax, gt_param, est_param, wind in zip(axarr[2*len(wind_vec):], curr_gt_A, curr_A, wind_vec):
            max_val = 100 #max(gt_param.max(), est_param.max())
            min_val = 80 #min(gt_param.min(), est_param.min())
            ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
            # ax.set_title(f'AoLP \n speed=' + str(wind) + '[m/s]')
            ax.set_title(f'AoLP \n albedo=' + str(wind))
            ax.set_xlim([0.9 * min_val, 1.1 * max_val])
            ax.set_ylim([0.9 * min_val, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)
            ax.set_aspect('equal')
        # fig.suptitle('image #'+str(im_ind)+' scatter plots for different wind speeds', size=16, y=0.95)
        fig.suptitle('image #' + str(im_ind) + ' scatter plots for different albedos', size=16, y=0.95)

        # fig, axarr = plt.subplots(1, 3, figsize=(20, 20))
        # fig.subplots_adjust(hspace=0.2, wspace=0.2)
        # axarr = axarr.flatten()
        # axarr[0].plot(wind_vec, curr_delta_I)
        # axarr[0].set_title('I mass error values as a function of wind speed')
        # axarr[0].set_ylabel('mass error value', fontsize=14)
        # axarr[0].set_xlabel('wind speed [m/s]', fontsize=14)
        # axarr[1].plot(wind_vec, curr_delta_D)
        # axarr[1].set_title('DoLP mass error values as a function of wind speed')
        # axarr[1].set_ylabel('mass error value', fontsize=14)
        # axarr[1].set_xlabel('wind speed [m/s]', fontsize=14)
        # axarr[2].plot(wind_vec, curr_delta_A)
        # axarr[2].set_title('AoLP mass error values as a function of wind speed')
        # axarr[2].set_ylabel('mass error value', fontsize=14)
        # axarr[2].set_xlabel('wind speed [m/s]', fontsize=14)
        # fig.suptitle('image #' + str(im_ind) + ' mass error values', size=16, y=0.95)
    plt.show()




def AirMSPI_Test(test_ind=5, surface_wind_speed=10., render_cloud=True, zeta=None):
    # wind: 9.9-10.2m/s according to worldview.earthdata.
    stokes_list = ['I', 'Q', 'U']
    path_retrieval = "/wdata_visl/inbalkom/NN_outputs/test/2024-02-29/17-20-13-AirMSPI-dropped5python/airmspi_recovery_bomex.mat"
    #"/wdata_visl/inbalkom/NN_outputs/test/2024-02-29/16-04-16-AirMSPI-dropped5python/airmspi_recovery_bomex.mat"
                     # "/wdata/inbalkom/NN_outputs/AirMSPI/test_results/2023-12-07/15-02-30/airmspi_recovery.mat"
    mask_path = "/wdata/inbalkom/NN_Data/AIRMSPI_MEASUREMENTS/mask_72x72x32_vox50x50x40m.mat"

    output_path = '/'.join(path_retrieval.split('/')[:-1])
    output_path = '/'.join(output_path.split('/')[-3::])
    output_path = os.path.join('/wdata_visl/inbalkom/NN_Data/AirMSPI_NN_results/', output_path)

    if not os.path.exists(output_path):
        # Create a new directory because it does not exist
        safe_mkdirs(output_path)
        print("The directory for saving AirMSPI test images was created.")

    cloud_scatterer, mask = load_from_airmspi_mat(path_retrieval, mask_path, density='lwc')

    gt_meas_filename = '/wdata/inbalkom/AT3D_CloudCT_shared_files/AirMSPI_660_PySHDOM_Meas_dict_for_AT3D.pkl'
    with open(gt_meas_filename, 'rb') as f:
        gt_meas_data = pickle.load(f)

    sun_azimuth = np.mean(gt_meas_data['_sun_azimuth_list'])
    sun_zenith = np.mean(gt_meas_data['_sun_zenith_list'])

    if render_cloud:
        # make sure all values will exist in the mie tables
        cloud_scatterer.veff.data[cloud_scatterer.veff.data <= 0.02] = 0.0201
        cloud_scatterer.veff.data[cloud_scatterer.veff.data >= 0.55] = 0.55
        cloud_scatterer.reff.data[cloud_scatterer.reff.data <= 0.01] = 0.0101
        cloud_scatterer.reff.data[cloud_scatterer.reff.data >= 35] = 35 - 1.1e-3
    else:
        cloud_scatterer.density.data[:, :, :] = 0.
        cloud_scatterer.density.data[1, 1, 1] = 0.1
        cloud_scatterer.veff.data[:, :, :] = 0.0201
        cloud_scatterer.reff.data[:, :, :] = 0.0101
    # load atmosphere
    atmosphere = xr.open_dataset('../data/ancillary/AFGL_summer_mid_lat.nc')
    # subset the atmosphere, choose only the bottom twenty km.
    reduced_atmosphere = atmosphere.sel({'z': atmosphere.coords['z'].data[atmosphere.coords['z'].data <= 20.0]})
    # merge the atmosphere and cloud z coordinates
    merged_z_coordinate = at3d.grid.combine_z_coordinates([reduced_atmosphere, cloud_scatterer])

    # define the property grid - which is equivalent to the base RTE grid
    rte_grid = at3d.grid.make_grid(cloud_scatterer.x.diff('x')[0], cloud_scatterer.x.data.size,
                                   cloud_scatterer.y.diff('y')[0], cloud_scatterer.y.data.size,
                                   merged_z_coordinate)

    cloud_scatterer_on_rte_grid = at3d.grid.resample_onto_grid(rte_grid, cloud_scatterer)

    size_distribution_function = at3d.size_distribution.gamma

    ##### get optical property generator #####
    wavelength_bands = [[gt_meas_data['_wavelength'], gt_meas_data['_wavelength']]]
    mean_wavelengths = [np.mean(wavelength_band) for wavelength_band in wavelength_bands]

    mie_mono_tables = OrderedDict()
    for mean_wavelength, wavelength_band in zip(mean_wavelengths, wavelength_bands):
        wavelength_band_tuple = (wavelength_band[0], wavelength_band[1])

        wavelen1, wavelen2 = wavelength_band_tuple

        if wavelen1 == wavelen2:
            wavelength_averaging = False
            formatstr = 'mie_mono_Water_{}nm.nc'.format(int(1e3 * wavelen1))
        else:
            wavelength_averaging = True
            formatstr = 'mie_mono_averaged_Water_{}-{}nm.nc'.format(int(1e3 * wavelength_band[0]),
                                                                    int(1e3 * wavelength_band[1]))
        mono_path = os.path.join('/wdata/inbalkom/AT3D_CloudCT_shared_files/mie_tables', formatstr)
        mie_mono_table = at3d.mie.get_mono_table(
            'Water', wavelength_band_tuple,
            wavelength_averaging=wavelength_averaging,
            max_integration_radius=65.0,
            minimum_effective_radius=0.1,
            relative_dir='/wdata/inbalkom/AT3D_CloudCT_shared_files/mie_tables/',
            verbose=False
        )
        mie_mono_tables[mean_wavelength] = mie_mono_table
        print('added mie with wavelength of {}'.format(mean_wavelength))
        # mie_mono_table.to_netcdf(mono_path)

    optical_property_generator = at3d.medium.OpticalPropertyGenerator(
        'cloud',
        mie_mono_tables,
        size_distribution_function,
        reff=np.linspace(0.01, 35.0, 30),
        veff=np.linspace(0.02, 0.56, 10),
    )
    optical_properties = optical_property_generator(cloud_scatterer_on_rte_grid)

    # one function to generate rayleigh scattering.
    rayleigh_scattering = at3d.rayleigh.to_grid(mean_wavelengths, atmosphere, rte_grid)

    solvers_dict = at3d.containers.SolversDict()
    # note we could set solver dependent surfaces / sources / numerical_config here
    # just as we have got solver dependent optical properties.
    # surface = at3d.surface.lambertian(surface_wind_speed)
    if zeta==None:
        surface = at3d.surface.wave_fresnel(real_refractive_index=1.331, imaginary_refractive_index=2e-8,
                                            surface_wind_speed=surface_wind_speed,
                                            ground_temperature=288.15)  # 15 degrees Celcius
        print('added wave_fresnel surface with const wind speed of {} m/s'.format(surface_wind_speed))
    else:
        sigma = np.sqrt((0.003+0.00512*surface_wind_speed)/2)
        surface = at3d.surface.diner(A=0.04, K=0.8, B=0.4, ZETA=zeta, SIGMA=sigma, ground_temperature=288.15)  # 15 degrees Celcius
        print('added Diner et al. surface with const wind speed of {} m/s zeta of {}'.format(surface_wind_speed, zeta))



    for wavelength in mean_wavelengths:
        medium = {
            'cloud': optical_properties[wavelength],
            'rayleigh': rayleigh_scattering[wavelength]
        }
        config = at3d.configuration.get_config()
        solvers_dict.add_solver(
            wavelength,
            at3d.solver.RTE(
                numerical_params=config,
                surface=surface,
                source=at3d.source.solar(wavelength, np.cos(sun_zenith * np.pi / 180), sun_azimuth),
                medium=medium,
                num_stokes=3
            )

        )

    solvers_dict.solve(n_jobs=60, maxiter=150)

    ##### define sensors #####

    # Correct measurments projection x and y so that the projections will
    # intersect in a domain around a center of the medium. This medium
    # and the corrections are always set manually for each cloud.
    CORR_PROJ_BOMEX = True

    sensor_dict = at3d.containers.SensorsDict()
    names = []
    image_shape = np.array([350, 350])
    projection = copy.deepcopy(gt_meas_data['_camera']['_projection'])
    if CORR_PROJ_BOMEX:
        for i, pro in enumerate(projection['_projection_list']):
            pro['_x'] += 0.6 + (0.05 * 20)
            pro['_y'] += 0.75 + (0.05 * 20)
            sensor = at3d.sensor.make_sensor_dataset(
                pro['_x'], pro['_y'], pro['_z'], pro['_mu'], pro['_phi'],
                stokes_list, gt_meas_data['_wavelength'], fill_ray_variables=True)
            sensor['image_shape'] = xr.DataArray(image_shape, coords={'image_dims': ['nx', 'ny']}, dims='image_dims')
            instrument_name = str(int(gt_meas_data['_wavelength'] * 1000))
            sensor_dict.add_sensor(instrument_name, sensor)
            names.append('proj'+str(i))
    else:
        for i, pro in enumerate(projection['_projection_list']):
            sensor = at3d.sensor.make_sensor_dataset(
                pro['_x'], pro['_y'], pro['_z'], pro['_mu'], pro['_phi'],
                stokes_list, gt_meas_data['_wavelength'], fill_ray_variables=True)
            sensor['image_shape'] = xr.DataArray(image_shape, coords={'image_dims': ['nx', 'ny']}, dims='image_dims')
            instrument_name = str(int(gt_meas_data['_wavelength'] * 1000))
            sensor_dict.add_sensor(instrument_name, sensor)
            names.append('proj' + str(i))


    print('Done defining AIRMSPI''s sensors')
    print('getting AIRMSPI''s measurments')
    # Next part will be the rendering, when the RTE solver is prepared (below).
    # get the measurements
    sensor_dict.get_measurements(solvers_dict, n_jobs=60, verbose=True)
    print('Done getting AIRMSPI''s measurments')

    # add AirMSPI noise to rendered images:
    sensor_dict, _ = add_airmspi_noise(sensor_dict, stokes_list, names)

    gt_images = gt_meas_data['_images_nomask']

    images = []
    images_scatter = []
    gt_images_scatter = []
    for instrument_ind, (instrument, sensor_group) in enumerate(sensor_dict.items()):
        sensor_images = sensor_dict.get_images(instrument)
        sensor_group_list = sensor_dict[instrument]['sensor_list']
        assert len(names) == len(sensor_group_list), "len(names) does not match len(sensor_group_list)"
        for sensor_ind, (sensor, sensor_name) in enumerate(zip(sensor_group_list, names)):
            if (stokes_list == ['I']) or (stokes_list == 'I'):
                curr_image = np.array([sensor_images[sensor_ind].I.data])
            else:
                curr_image = np.dstack([sensor_images[sensor_ind][pol_channel].data for pol_channel in stokes_list])
                # curr_image = np.stack(
                #     [sensor_images[sensor_ind][pol_channel].data for pol_channel in stokes_list])
            images.append(np.transpose(curr_image,[2, 0, 1]))
            copied = sensor.copy(deep=True)

            images_scatter.append(calc_image_in_scattering_plane_vectorbase(copied, curr_image, sensor_name, sun_azimuth, sun_zenith))
            assert np.allclose(images[-1][0], images_scatter[-1][0]), "Bad calculation of scattering plane."
            gt_images_scatter.append(calc_image_in_scattering_plane_vectorbase(copied, gt_images[sensor_ind], sensor_name, sun_azimuth, sun_zenith))
            assert np.allclose(gt_images[sensor_ind][:,:,0], gt_images_scatter[-1][0]), "Bad calculation of scattering plane."

    gt_images = [np.transpose(gt_im, [2, 0, 1]) for gt_im in gt_images]

    DOLP_scatter = [np.sqrt(im[1]**2+im[2]**2)/im[0] for im in images_scatter]
    AOLP_scatter = [np.rad2deg(0.5*np.arctan2(im[2], im[1])) for im in images_scatter]
    for im in AOLP_scatter:
        im[im<0]=im[im<0]+180
    gt_DOLP_scatter = [np.sqrt(im[1] ** 2 + im[2] ** 2) / (im[0] + 1e-6) for im in gt_images_scatter]
    gt_AOLP_scatter = [np.rad2deg(0.5 * np.arctan2(im[2], im[1])) for im in gt_images_scatter]
    for im in gt_AOLP_scatter:
        im[im < 0] = im[im < 0] + 180
    if 0:
        plot_cloud_images(images)
        plot_cloud_images(gt_images)
        show_scatter_plot_colorbar(gt_images[test_ind][0,80:300,:], images[test_ind][0,80:300,:], param_name='I, wind={}'.format(surface_wind_speed), pixel_precent=0.01)
        show_scatter_plot_colorbar(gt_images[test_ind][1,80:300,:], images[test_ind][1,80:300,:], param_name='Q mer, wind={}'.format(surface_wind_speed), pixel_precent=0.01)
        show_scatter_plot_colorbar(gt_images[test_ind][2,80:300,:], images[test_ind][2,80:300,:], param_name='U mer, wind={}'.format(surface_wind_speed), pixel_precent=0.01)
        show_scatter_plot_colorbar(gt_DOLP_scatter[test_ind][80:300, :], DOLP_scatter[test_ind][80:300, :], param_name='DoLP, wind={}'.format(surface_wind_speed), pixel_precent=0.01)
        show_scatter_plot_colorbar(gt_AOLP_scatter[test_ind][80:300, :], AOLP_scatter[test_ind][80:300, :], param_name='AoLP scattering, wind={}'.format(surface_wind_speed), pixel_precent=0.01)
        a = 5

    gt_I = np.array(gt_images)[:, 0, 0:24, 100:200]
    I = np.array(images)[:, 0, 150:174, 100:200]
    gt_DoLP = np.array(gt_DOLP_scatter)[:, 0:24, 100:200]
    DoLP = np.array(DOLP_scatter)[:, 150:174, 100:200]
    gt_AoLP_scat = np.array(gt_AOLP_scatter)[:, 0:24, 100:200]
    AoLP_scat = np.array(AOLP_scatter)[:, 150:174, 100:200]
    print("--------------")
    return gt_I, I, gt_DoLP, DoLP, gt_AoLP_scat, AoLP_scat



def plot_cloud_images(images):
    # ------------------
    # I:
    fig, axarr = plt.subplots(3, 3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axarr = axarr.flatten()
    vmin = np.array(images)[:, 0, :, :].min()
    vmax = np.array(images)[:, 0, :, :].max()
    for ind, (ax, image) in enumerate(zip(axarr, images)):
        image = np.squeeze(image.copy())
        im = ax.imshow(image[0, ...], cmap='gray', vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01)
        plt.colorbar(im, cax=cax)
        ax.set_title('image #'+str(ind))
    fig.suptitle('I', size=16, y=0.95)

    # ------------------
    # Q:
    fig, axarr = plt.subplots(3, 3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axarr = axarr.flatten()
    vmin = np.array(images)[:, 1, :, :].min()
    vmax = np.array(images)[:, 1, :, :].max()
    for ax, image in zip(axarr, images):
        image = np.squeeze(image.copy())
        im = ax.imshow(image[1, ...], cmap='gray', vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01)
        plt.colorbar(im, cax=cax)
    fig.suptitle('Q', size=16, y=0.95)

    # ------------------
    # U:
    fig, axarr = plt.subplots(3, 3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axarr = axarr.flatten()
    vmin = np.array(images)[:, 2, :, :].min()
    vmax = np.array(images)[:, 2, :, :].max()
    for ax, image in zip(axarr, images):
        image = np.squeeze(image.copy())
        im = ax.imshow(image[2, ...], cmap='gray', vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01)
        plt.colorbar(im, cax=cax)
    fig.suptitle('U', size=16, y=0.95)

    # --------------------
    plt.show()
    # mlab.show()
    print('done plotting')


if __name__ == '__main__':
    main()
