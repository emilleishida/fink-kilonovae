# Copyright 2021 Fink Software
# Author: Emille E. O. Ishida and Biswajit Biswas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import argparse


def fluxcal2mag(fluxcal):
    """Convert FLUXCAL to magnitude.
    
    Parameters
    ----------
    fluxcal: float or np.array
    
    Returns
    -------
    magnitude: float or np.array
    """
    
    mag = 2.5 * (11 - np.log10(fluxcal))
    
    return mag

def filter_points(obs_mjd: np.array, obs_flux: np.array, 
                  PC_epoch_grid: np.array):
    """Translate observed points to an epoch grid to match the PCs.
    
    Parameters
    ----------
    obs_mjds: np.array
        Values for observed mjds.
    obs_flux: np.array
        Values for fluxes at observed mjds.
    PC_epoch_grid: np.array
        Values of epochs grid used in constructing the PCs.
        Time bin between each entry should be the same.
        
    Returns
    -------
    new_mjd: np.array
        Values of mjds compatible to observations and PCs.
    new_flux: np.array
        Values of flux for each new_mjd. 
        If more than one observation is available in a time bin
        this corresponds to the mean of all observations within 
        the bin.    
    mjd_flag: np.array of bool
        Mask for PC_epoch_grid filtering only allowed MJDs.
    mjd_cent: float
        Centered MJD value.
    """
    
    flux_final = []
    mjd_flag = []
    
    # get time bin
    time_bins = [np.round(PC_epoch_grid[i + 1] - PC_epoch_grid[i], 3) 
                 for i in range(1, len(PC_epoch_grid) - 1)]

    if np.unique(time_bins).shape[0] > 1:
        raise ValueError('PC_epoch_grid should have uniform binning.')
        
    else:
        time_bin = np.unique(time_bins)[0]
        
    # determine time of maximum    
    mjd_cent = obs_mjd[list(obs_flux).index(max(obs_flux))]
    
    # centralize epochs
    epochs = obs_mjd - mjd_cent
    
    # identify which bins have measurements
    for i in range(PC_epoch_grid.shape[0]):
        flag1 = epochs >= PC_epoch_grid[i] - 0.5 * time_bin
        flag2 = epochs < PC_epoch_grid[i] + 0.5 * time_bin
        flag3 = np.logical_and(flag1, flag2)
    
        # keep rising epochs with measurements
        if sum(flag3) > 0 and PC_epoch_grid[i] <= 0:
            flux_final.append(np.mean(obs_flux[flag3]))
            mjd_flag.append(True)
        else:
            mjd_flag.append(False)
    
    # populate epoch bins with available measurements
    if sum(mjd_flag) > 0:
        mjd_flag = np.array(mjd_flag)
    
        new_mjd = PC_epoch_grid[mjd_flag]
        new_flux = np.array(flux_final)
    
        return new_mjd, new_flux, mjd_flag, mjd_cent
    
    else:
        return [], [], None, None
    
    
def extract_features(mjd: np.array, flux: np.array, epoch_lim: list,
                     time_bin:float, pcs: pd.DataFrame, 
                     flux_lim=0, phot=False):
    """
    Extract features from light curve.
    
    Parameters
    ----------
    mjd: np.array
        Values for MJD.
    flux: np.array
        Values for FLUXCAL.
    epoch_lim: list
        Min and max epoch since maximum brightness to consider.
        Format is [lower_lim, upper_lim]. 
    time_bin: float
        Width of time gap between two elements in PCs.
    pcs: pd.DataFrame
        All principal components to be considered.
        keys should be PCs names (1, 2, 3, ...), 
        values their amplitude at each epoch in the grid.
    flux_lim: float (optional)
        Min flux cut applied to all points. Default is 0.
    phot: bool (optional)
        If True return surviving photometric grid points. 
        Default is False.
        
    Returns
    -------
    features: np.array
        Features for this light curve. Order is:
        [n_points, residual_from_fit, coefficients, max_flux]
    phot: dict (optional)
        Keywords are [MJD, FLUXCAL] in the same bins defined
        by the PCs and MJD flag.
    """
    
    # create list for storing output
    cut_data = []
    features = []
    rec = []
    mjd0 = None
    
    # get useful flux
    flux_flag = flux >= flux_lim
    
    # construct epoch grid
    PC_epoch_grid = np.arange(epoch_lim[0], epoch_lim[1] + time_bin, time_bin)

    if sum(flux_flag) > 0:
        
        max_flux = max(flux[flux_flag])
        
        # translate point to suitable grid
        new_mjd, new_flux, mjd_flag, mjd0 = \
            filter_points(obs_mjd=mjd, obs_flux=flux, 
                          PC_epoch_grid=PC_epoch_grid)
        
        coef_mat = pd.DataFrame()
        for key in pcs.keys():
            coef_mat[key] = pcs[key].values[mjd_flag]

        # fit coefficients
        max_newflux = max(new_flux)
            
        x, res, rank, s = np.linalg.lstsq(coef_mat.values, 
                                          new_flux/max_newflux,
                                          rcond=None)

        # add number of points and residuals and 
        # coefficients to the matrix
        features.append(len(new_mjd))
                
        if len(res) > 0:
            features.append(res[0])
        else:
            features.append(-9)
                
        for elem in x:
            features.append(elem)
            
        features.append(max_newflux)
        
        # return photometry for posterior color calculations
        obs_phot = {}
        obs_phot['new_mjd'] = new_mjd
        obs_phot['new_flux'] = new_flux
        obs_phot['mjd_flag'] = mjd_flag
            
    else:
        features = [0 for i in range(len(pcs.keys()) + 3)]

        obs_phot = {}
        obs_phot['new_mjd'] = None
        obs_phot['new_flux'] = None
        obs_phot['mjd_flag'] = None
    
    return features, obs_phot

def extract_all_filters(epoch_lim: list, pcs: pd.DataFrame, 
                        time_bin: float, filters: list, 
                        lc: pd.DataFrame, flux_lim=0, colors=False):
    """Extract features from 1 object in all available filters.
    
    Parameters
    ----------
    epoch_lim: list
        Min and max epoch since maximum brightness to consider.
        Format is [lower_lim, upper_lim]. 
    filters: list
        List of broad band filters.
    lc: pd.DataFrame
        Keys should be ['MJD', 'FLUXCAL', 'FLT'].         
    pcs: pd.DataFrame
        All principal components to be considered.
        keys should be PCs names (1, 2, 3, ...), 
        values their amplitude at each epoch in the grid.
        Order of PCs when calling pcs.keys() is important.
    time_bin: float
        Width of time gap between two elements in PCs.
    colors: bool (optional)
        If True, the available colour measurement closest to max.
    flux_lim: float (optional)
        Min flux cut applied to all points. Default is 0.
    
    Returns
    -------
    all_features: list
        List of features for this object.
        Order is all features from first filter, then all features
        from second filters, etc.
    """
    
    # build epoch grid
    PC_epoch_grid = np.arange(epoch_lim[0], 
                              epoch_lim[1] + time_bin, time_bin)

    # store results from extract_features
    all_features = {}

    for fil in filters:
        all_features[fil] = {}
        
        filter_flag = lc['FLT'].values == fil
        
        # get number of surviving points
        npoints = sum(filter_flag)
    
        if npoints > 0:
            obs_mjd = lc['MJD'].values[filter_flag]
            obs_flux = lc['FLUXCAL'].values[filter_flag]
            
            # extract features
            res, phot = extract_features(mjd=obs_mjd, 
                                         flux=obs_flux,
                                         epoch_lim=epoch_lim,
                                         time_bin=time_bin,
                                         pcs=pcs,
                                         flux_lim=flux_lim,
                                         phot=colors)
            
            all_features[fil]['feat'] = res
            all_features[fil]['phot'] = phot
        else:
            all_features[fil]['feat'] = [None]
            all_features[fil]['phot'] = [None]
    
    # check if there is photometry allowing color calculation
    c = None
    if colors and len(all_features[filters[0]]['phot']) > 0 and \
       len(all_features[filters[1]]['phot']) > 0:
        # check if there are the same time bins in both filters
        bin_flag = np.logical_and(all_features[filters[0]]['phot']['mjd_flag'],
                                  all_features[filters[1]]['phot']['mjd_flag'])
        
        if sum(bin_flag) > 0:
            surv_epoch = max(PC_epoch_grid[bin_flag])
            
            mag = []
            for f in filters: 
                flag = all_features[f]['phot']['new_mjd'] == surv_epoch
                flux = all_features[f]['phot']['new_flux'][flag]
                mag.append(fluxcal2mag(flux))
                
            c = [mag[i] - mag[i + 1] for i in range(len(filters) - 1)]    
            
    # concatenate the available features
    matrix_line = []
    for fil in filters:
        matrix_line = matrix_line + list(all_features[fil]['feat'])
            
    # concatenate color in the epoch closest to maximum
    if c:
        matrix_line = matrix_line + list(c[0])
    else:
        matrix_line = matrix_line + [None]
            
    return matrix_line

def build_feature_matrix(epoch_lim: list, filters: list, 
                         header: pd.DataFrame, pcs: pd.DataFrame,
                         photo:pd.DataFrame, time_bin:float,
                         flux_lim=0, colors=False):
    """Build feature matrix for set of objects.
    
    Parameters
    ----------    
    epoch_lim: list
        Min and max epoch since maximum brightness to consider.
        Format is [lower_lim, upper_lim].
    filters: list
        List of broad band filters.
    header: pd.DataFrame
        Header information extracted from SNANA files.
    pcs: pd.DataFrame
        All principal components to be considered.
        keys should be PCs names (1, 2, 3, ...), 
        values their amplitude at each epoch in the grid.
    photo: pd.DataFrame
        Light curve information extracted from SNANA files.
    time_bin: float
        Width of time gap between two elements in PCs.
    colors: bool (optional)
        If True, add extra element with color at max.
        Default is False.
    flux_lim: float (optional)
        Min flux cut applied to all points. Default is 0.    
    
    Returns
    -------
    matrix: pd.DataFrame
        Complete feature matrix.
    """
    
    matrix = []
    
    for indx in range(header.shape[0]):
        
        print('*** ', indx, ' ***')
        
        snid = header['SNID'].values[indx]
        vartype = int(header['SIM_TYPE_INDEX'].values[indx])
        
        snid_flag = photo['SNID'].values == snid
        
        lc = photo[snid_flag]
    
        line = extract_all_filters(epoch_lim=epoch_lim, pcs=pcs, 
                                   time_bin=time_bin, filters=filters, 
                                   lc=lc, flux_lim=flux_lim,
                                   colors=colors)
        line.insert(0, vartype)
        line.insert(0, snid)
        
        matrix.append(line)
        
    matrix = np.array(matrix)
    
    # columns names
    names_root = ['npoints_', 'residuo_'] + \
                 ['coeff' + str(i + 1) + '_' 
                  for i in range(len(pcs.keys()))] + ['maxflux_']
    
    columns = []
    
    for f in filters:
        fname = ''.join(x for x in f if x.isalpha())
        
        for name in names_root:
            if len(fname.split()) > 1:
                columns.append(name + fname[1])
            else:
                columns.append(name + fname)
                
    columns.insert(0, 'type')
    columns.insert(0, 'SNID')
    columns.append('color')
    
    col = []
    for item in columns:
        col.append(item.replace('b', ''))
            
    return pd.DataFrame(matrix, columns=col)

def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
    """ Conversion from magnitude to Fluxcal from SNANA manual
    Parameters
    ----------
    magpsf: float
        PSF-fit magnitude from ZTF
    sigmapsf: float
    Returns
    ----------
    fluxcal: float
        Flux cal as used by SNANA
    fluxcal_err: float
        Absolute error on fluxcal (the derivative has a minus sign)
    """
    if magpsf is None:
        return None, None
    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10 ** 10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err

def extract_features_grandma(hf, epoch_lim: list,
                            time_bin:float, pcs: pd.DataFrame, 
                            flux_lim=0, phot=False):
    """Extract features from a set of GRANDMA simulated light curves.
    
    Parameters
    ----------
    hf: hdf5
        HDF5 object read from file.
    epoch_lim: list
        Min and max epoch since maximum brightness to consider.
        Format is [lower_lim, upper_lim]. 
    time_bin: float
        Width of time gap between two elements in PCs.
    pcs: pd.DataFrame
        All principal components to be considered.
        keys should be PCs names (1, 2, 3, ...), 
        values their amplitude at each epoch in the grid.
    flux_lim: float (optional)
        Min flux cut applied to all points. Default is 0.
    phot: bool (optional)
        If True return surviving photometric grid points. 
        Default is False.
        
    Returns
    -------
    matrix: np.array
        Features for all light curves. Column is:
        [n_points, residual_from_fit, coefficients, max_flux]
    """

    filters = ['g', 'r']
    
    m1 = []
    
    keys = list(hf.keys())

    for key in keys:
        
        line = []
        all_features = {}
        
        for f in filters: 
            all_features[f] = {}
            mag = hf.get(key).get(f).get('mag')[()]
            time = hf.get(key).get(f).get('time')[()]
            error = hf.get(key).get(f).get('magerr')[()]
            
            if len(mag) > 0:
                fluxcal, fluxcalerr = mag2fluxcal_snana(mag, error)

                res, photom = extract_features(mjd=time, flux=fluxcal, 
                                               epoch_lim=epoch_lim,
                                               time_bin=time_bin, pcs=pcs, 
                                               flux_lim=flux_lim, phot=phot)
            
                all_features[fil]['feat'] = res
                all_features[fil]['phot'] = photom
            else:
                all_features[fil]['feat'] = [None]
                all_features[fil]['phot'] = [None]
        
        # check if there is photometry allowing color calculation
        c = None
        if colors and len(all_features[filters[0]]['phot']) > 0 and \
           len(all_features[filters[1]]['phot']) > 0:
            # check if there are the same time bins in both filters
            bin_flag = np.logical_and(all_features[filters[0]]['phot']['mjd_flag'],
                                      all_features[filters[1]]['phot']['mjd_flag'])
        
            if sum(bin_flag) > 0:
                surv_epoch = max(PC_epoch_grid[bin_flag])
            
                mag = []
                for f in filters: 
                    flag = all_features[f]['phot']['new_mjd'] == surv_epoch
                    flux = all_features[f]['phot']['new_flux'][flag]
                    mag.append(fluxcal2mag(flux))
                
                c = [mag[i] - mag[i + 1] for i in range(len(filters) - 1)]    
            
        # concatenate the available features
        matrix_line = []
        for fil in filters:
            matrix_line = matrix_line + list(all_features[fil]['feat'])
            
        # concatenate color in the epoch closest to maximum
        if c:
            matrix_line = matrix_line + list(c[0])
        else:
            matrix_line = matrix_line + [None]
        
        matrix_line.insert(0, 51)
        matrix_line.insert(0, key)
        
        m1.append(matrix_line)
    
    # columns names
    names_root = ['npoints_', 'residuo_'] + \
                 ['coeff' + str(i + 1) + '_' 
                  for i in range(len(pcs.keys()))] + ['maxflux_']
    
    columns = []
    
    for f in filters:
        fname = ''.join(x for x in f if x.isalpha())
          
        for name in names_root:
            if len(name.split()) > 1:
                columns.append(name + fname[-1])
            else:
                columns.append(name + fname)
               
    columns.insert(0, 'type')
    columns.insert(0, 'SNID')
    columns.append('color')

    matrix = pd.DataFrame(m1, columns=columns)
    
    return matrix



def main(user_input):
    """Built feature matrix.
    
    Parameters
    ----------
    -c: str
        Path to principal components file.
    -d: str
        Data set to perform feature extraction. Options are
        'Fink' or 'GRANDMA'.
    -e: list
        Min and Max epoch limits used to construct PCs.
    -n: int
        Numper of components to use.
    -o: str
        Path to output file to store matrix.
    -p: str
        Path to light curves file.   
    -t: float
        Time bin used to construct PCs.
    -f: float (optional)
        Minimum flux considered. Default is 0.
    -m: str (optional)
        Path to metadata (or header) file. Only used if
        dataset == 'Fink'. Default is None.
    -l: bool or int (optional)
        If True or > 0 add column with color at max.
    """
    
    # load components
    comp = pd.read_csv(user_input.comp_fname)
    
    # get only the required number of PCs in the correct format
    pcs = pd.DataFrame()
    for i in range(user_input.npcs):
        pcs[i + 1] = comp.iloc[i].values

    if user_input.dataset == 'Fink':
        # read data
        header = pd.read_csv(user_input.header_fname)
        photo = pd.read_csv(user_input.photo_fname)        
        
        # get filters names
        filters = np.unique(photo['FLT'].values)
    
        matrix = build_feature_matrix(epoch_lim=user_input.epoch_lim, 
                                  filters=filters, 
                                  header=header, pcs=pcs,
                                  photo=photo, time_bin=user_input.time_bin,
                                  flux_lim=user_input.flux_lim,
                                  colors=user_input.colors)
        
        matrix.to_csv(user_input.output_fname, index=False)
        
    elif user_input.dataset == 'GRANDMA':
        
        import h5py
        
        filters = ['g', 'r']
        
        hf = h5py.File(user_input.photo_fname)
        
        matrix = extract_features_grandma(hf, 
                                          epoch_lim=user_input.epoch_lim,
                                          time_bin=user_input.time_bin, 
                                          pcs=pcs, 
                                          flux_lim=user_input.flux_lim,
                                          phot=user_input.colors)
        
    # remove null coefficients
    zeros = {}
    for i in range(user_input.npcs):
        zeros[i] = np.logical_or(matrix['coeff' + str(i + 1) + '_g'].values == 0, 
                                 matrix['coeff' + str(i + 1) + '_r'].values == 0)
        
    z = np.array([False for i in range(user_input.npcs)])
    for j in range(len(zeros)):
        z = np.logical_or(z[j], zeros[j])
        
    matrix_clean = matrix[~z]
    matrix_clean['orig_sims'] = user_input.dataset
    matrix_clean.to_csv(user_input.output_fname, index=False)
    
    
if __name__ == '__main__':

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='Feature extraction for '
                                                 'the KN module')
    parser.add_argument('-n', '--n-components', dest='npcs',
                        required=True, type=int,
                        help='Number of principal components to use.')
    parser.add_argument('-e', '--epoch-limits-list', dest='epoch_lim',
                        required=True, type=int,
                        help='Min and Max epoch limits used to construct PCs.',
                        nargs='+')
    parser.add_argument('-m', '--metadata-fname', dest='header_fname',
                        required=False, type=str, default=None,
                        help='Path to metadata file.')
    parser.add_argument('-c', '--comp-fname', dest='comp_fname', 
                        required=True, type=str, 
                        help='Path to principal components file.')
    parser.add_argument('-d', '--data-set', dest='dataset',
                       required=True, type=str, 
                        help='Data set. Options are Fink or GRANDMA.')
    parser.add_argument('-o', '--output-fname', dest='output_fname',
                       required=True, type=str,
                       help='Output file name.')
    parser.add_argument('-p', '--photo-fname', dest='photo_fname',
                        required=True, type=str,
                       help='Path to light curves file.')
    parser.add_argument('-t', '--time-bin', dest='time_bin',
                        required=True, type=float,
                       help='Time bin used to construct PCs.')
    parser.add_argument('-f', '--min-flux-lim', dest='flux_lim',
                        required=False, type=float, default=0,
                       help='Minimum flux cut. Default is 0.')
    parser.add_argument('-l', '--color', dest='colors',
                       required=False, type=int, default=0,
                       help='Add column with color.')
    
    from_user = parser.parse_args()

    main(from_user)
