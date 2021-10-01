#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script interpolates various ocean/climate variables to the locations of
coral samples taken by April Burt
@author: Noam Vogt-Vincent

DATA SOURCES:
    SST: OSTIA Sea Surface Temperature and Sea Ice Analysis
         https://resources.marine.copernicus.eu/product-detail/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011/INFORMATION
         Processed with cdo setmisstodis to interpolate missing values

    SSS: GLORYS12V1
         https://resources.marine.copernicus.eu/product-detail/GLOBAL_REANALYSIS_PHY_001_030/INFORMATION
         Processed with cdo setmisstodis to interpolate missing values

    Waves: WAVERYS
           https://resources.marine.copernicus.eu/product-detail/GLOBAL_REANALYSIS_WAV_001_032/INFORMATION

    Turbidity: ZSD from Copernicus-GlobColour
               https://resources.marine.copernicus.eu/product-detail/OCEANCOLOUR_GLO_OPTICS_L4_REP_OBSERVATIONS_009_081/INFORMATION
               Processed with cdo setmisstodis to interpolate missing values

    Surface currents: WINDS

    Tidal range: TPXO9v2a
                 https://www.tpxo.net/global/tpxo9-atlas

    All statistics are calculated from DAILY MEANS where relevant

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

##############################################################################
# DIRECTORIES ################################################################
##############################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = this_dir + '/data/'

# DATA
data_fh = {'sss'       : data_dir + 'sss/GLOBAL_REANALYSIS_PHY_001_030_processed.nc',
           'sss_mc'    : data_dir + 'sss/GLOBAL_REANALYSIS_PHY_001_030_monclim_processed.nc',
           'sst'       : data_dir + 'sst/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011_processed.nc',
           'sst_mc'    : data_dir + 'sst/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011_monclim_processed.nc',
           'zsd'       : data_dir + 'zsd/OCEANCOLOUR_GLO_OPTICS_L4_REP_OBSERVATIONS_009_081-TDS_processed.nc',
           'zsd_mc'    : data_dir + 'zsd/OCEANCOLOUR_GLO_OPTICS_L4_REP_OBSERVATIONS_009_081-TDS_monclim_processed.nc',
           'wav'       : data_dir + 'wav/GLOBAL_REANALYSIS_WAV_001_032-TDS.nc',
           'wav_mc'    : data_dir + 'wav/GLOBAL_REANALYSIS_WAV_001_032-TDS_monclim.nc',
           'tide_h'    : data_dir + 'tides/h_tpxo9.v2a.nc',}

var_names = {'sss' : 'so',
             'sst' : 'analysed_sst',
             'zsd' : 'ZSD',
             'wav' : 'VHM0'}

lon_names = {'sss' : 'longitude',
             'sst' : 'lon',
             'zsd' : 'lon',
             'wav' : 'longitude'}

lat_names = {'sss' : 'latitude',
             'sst' : 'lat',
             'zsd' : 'lat',
             'wav' : 'latitude'}

##############################################################################
# SCRIPT #####################################################################
##############################################################################

def nearest_idx(arr, val):
    # Function to find the nearest grid point
    idx = (np.abs(arr - val)).argmin()
    return idx

# Firstly import coral sites, clean data, and convert dms -> dd
site_data = pd.read_csv(data_dir+'coral_site_list.csv')
site_data['Lat'] = site_data['Lat'].str.replace(r' ', '')
site_data['Lon'] = site_data['Lon'].str.replace(r' ', '')

nsites = len(site_data['Lat'])

loc = {'lon' : np.zeros([nsites,], dtype=np.float32),
       'lat' : np.zeros([nsites,], dtype=np.float32)}

for site in range(nsites):
    lon_str = site_data['Lon'][site]
    lat_str = site_data['Lat'][site]

    def dms2dd(dms):
        dd_deg = lambda dms : int(dms.split('°', 1)[0])
        dd_min = lambda dms : int((dms.split('°', 1)[1]).split('\'')[0])
        dd_sec = lambda dms : float((dms.split('\"', 2)[0]).split('\'')[1])
        dd_sig = lambda dms : dms[-1]

        dd = np.float32(dd_deg(dms))
        dd += dd_min(dms)/60
        dd += dd_sec(dms)/3600

        if dd_sig(dms) in ['W', 'S']:
            dd *= -1

        return dd

    loc['lon'][site] = dms2dd(site_data['Lon'][site])
    loc['lat'][site] = dms2dd(site_data['Lat'][site])

# Now create arrays to hold the interpolated data (each bottom-level array will
# contain a N x 1 vector for the N sites)

clim = {'sst' : [],
        'sss' : [],
        'wav' : [],
        'zsd' : [],
        'vel' : []}

for key in clim:
    clim[key] = {'total_mean'    : np.zeros([nsites,], dtype=np.float32),
                 'total_std'     : np.zeros([nsites,], dtype=np.float32),
                 'total_min'     : np.zeros([nsites,], dtype=np.float32),
                 'total_max'     : np.zeros([nsites,], dtype=np.float32),
                 'monclim'       : np.zeros([nsites,12], dtype=np.float32),
                 'monclim_min'   : np.zeros([nsites,], dtype=np.float32),
                 'monclim_max'   : np.zeros([nsites,], dtype=np.float32)}

clim['tid'] = {'range' : np.zeros([nsites,], dtype=np.float32)}

# TIDES
# Methodology:
# Construct a 5-year time-series and calculate tidal range from MHW - MLW with
# a nearest-neighbour interpolation

with Dataset(data_fh['tide_h'], mode='r') as nc:
    tide_lon = nc.variables['lon_z'][:, 0]
    tide_lat = nc.variables['lat_z'][0, :]

    tide_amp = nc.variables['ha'][:]
    tide_pha = nc.variables['hp'][:]
    tide_con = nc.variables['con'][:]
    tide_per = np.array([12.4206012,
                         12.0,
                         12.65834751,
                         11.96723606,
                         23.93447213,
                         25.81933871,
                         24.06588766,
                         26.868350])
    tide_frq = 2*np.pi/tide_per

    tide_time = np.arange(0, 24*360*5, step=0.25)

for site in range(nsites):
    # Find location
    lon = loc['lon'][site]
    lat = loc['lat'][site]

    # Find nearest grid point
    lon_idx = nearest_idx(tide_lon, lon)
    lat_idx = nearest_idx(tide_lat, lat)

    amp = tide_amp[:, lon_idx, lat_idx]
    pha = tide_pha[:, lon_idx, lat_idx]

    tide_h = np.zeros_like(tide_time)

    for const in range(8):
        # Calculate SLH for the first 8 constituents for 5 years
        tide_h += amp[const]*np.cos((tide_frq[const]*tide_time) + pha[const])

    # Now calculate the daily maximum and minimum
    tide_h = tide_h.reshape((-1, 96)) # Each row = 1 day
    mhw = np.mean(np.amax(tide_h, axis=1))
    mlw = np.mean(np.amin(tide_h, axis=1))
    tide_range = mhw - mlw

    clim['tid']['range'][site] = tide_range

# Clear memory-intensive variables
del tide_lon, tide_lat, tide_amp, tide_pha, tide_time, tide_h

# Now calculate SSS, SST, ZSD, WAVES (same methodology)
for var in ['sst', 'sss', 'wav', 'zsd']:
    with Dataset(data_fh[var], mode='r') as nc:
        var_lon = nc.variables[lon_names[var]][:]
        var_lat = nc.variables[lat_names[var]][:]

        var_data = nc.variables[var_names[var]][:]

    with Dataset(data_fh[var + '_mc'], mode='r') as nc:
        var_clim_data = nc.variables[var_names[var]][:]

    for site in range(nsites):
        # Find location
        lon = loc['lon'][site]
        lat = loc['lat'][site]

        # Find nearest grid point
        lon_idx = nearest_idx(var_lon, lon)
        lat_idx = nearest_idx(var_lat, lat)

        if var in ['wav', 'zsd']:
            var_site = var_data[:, lat_idx, lon_idx]
            var_clim_site = var_clim_data[:, lat_idx, lon_idx]
        elif var == 'sst':
            var_site = var_data[:, lat_idx, lon_idx] - 273.15
            var_clim_site = var_clim_data[:, lat_idx, lon_idx] - 273.15
        elif var == 'sss':
            var_site = var_data[:, 0, lat_idx, lon_idx]
            var_clim_site = var_clim_data[:, 0, lat_idx, lon_idx]
        else:
            raise NotImplementedError('Variable type unknown.')

        # Check that cell is not masked
        if np.ma.is_masked(var_site[0]):
            raise NotImplementedError('Use cdo to interpolate masked sites!')

        clim[var]['total_mean'][site] = np.mean(var_site)
        clim[var]['total_std'][site] = np.std(var_site)
        clim[var]['total_min'][site] = np.min(var_site)
        clim[var]['total_max'][site] = np.max(var_site)
        clim[var]['monclim'][site, :] = var_clim_site
        clim[var]['monclim_min'][site] = np.min(var_clim_site)
        clim[var]['monclim_max'][site] = np.max(var_clim_site)

    del var_data, var_clim_data

# Now add results to the dataframe
for var in ['sst', 'sss', 'wav', 'zsd']:
    for stat in ['total_mean', 'total_std', 'total_min', 'total_max',
                 'monclim_min', 'monclim_max']:
        site_data[stat + '_' + var] = clim[var][stat][:]

    for i, mon in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                             'aug', 'sep', 'oct', 'nov', 'dec']):
        site_data[var + '_monclim_' + mon] = clim[var]['monclim'][:, i]

site_data['tidal_range'] = clim['tid']['range']

# Export to CSV
site_data.to_csv('coral_clim.csv', index=False)