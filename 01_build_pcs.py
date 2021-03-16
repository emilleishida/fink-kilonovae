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

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


def create_PCS(fname_header: str, fname_photo: str, npcs: int,
               epoch_lim: list, fname_output: str, mjd_gap=5, flux_lim=300h,
               time_bin=0.25):
    """Create principal components from a set of PERFECT simulations.
    
    Parameters
    ----------
    fname_header: str
        Path to header file.
    fname_photo: str
        Path to photo file.
    npcs: int
        Number of PCs to extract.
    epoch_lim: list
        Min and max epoch since maximum brightness to consider.
        Format is [lower_lim, upper_lim].
    fname_output: str
        Output file name.
    mjd_gap: int (optional)
        extra gap after epoch lim to extrapolate. Avoid boarder issues.
        Default is 5.
    flux_lim: float (optional)
        Minimum flux to avoid noise. Default is 2.5.
    time_bin: float (optional)
        Width of time bin between 2 columns in days. Default is 0.25.
        
    Returns
    -------
    Writes components to file.
    """

    # read header data
    header = pd.read_csv(fname_header)

    # read photo data
    photo = pd.read_csv(fname_photo)

    # get only u-band data
    u_flag = photo['FLT'].values == 'u'
    photo_u = photo[u_flag]

    # build light curve matrix
    full_matrix = []

    # days since max to give as output
    epoch_output = np.arange(epoch_lim[0], epoch_lim[1] + time_bin, time_bin)

    for sn in header['SNID'].values:
        # separate data for 1 object
        id_flag = photo['SNID'].values == sn
        lc_flag = np.logical_and(id_flag, u_flag)
    
        flux = photo['FLUXCAL'].values[lc_flag]
        fluxerr = photo['FLUXCALERR'].values[lc_flag]
        mjd = photo['MJD'].values[lc_flag]
    
        # centralize the data
        max_flux = max(flux)
    
        if max_flux > flux_lim:
            indx_maxflux = list(flux).index(max_flux)
            mjd0 = mjd[indx_maxflux]
            mjd_cent = mjd - mjd0
    
            mjd_min = min(mjd_cent)
            mjd_max = max(mjd_cent)
    
            # make sure data within required windown days of max
            if mjd_min <= epoch_lim[0] and mjd_max >= epoch_lim[1]:
                new_mjd = mjd_cent
                new_flux = flux
            elif mjd_min > epoch_lim[0] and mjd_max > epoch_lim[1]:
                new_mjd = [epoch_lim[0] - mjd_gap] + list(mjd_cent)
                new_flux = [0] + list(flux)
            elif mjd_min < epoch_lim[0] and mjd_max < epoch_lim[1]:
                new_mjd = list(mjd_cent) + [epoch_lim[1] + mjd_gap]
                new_flux = list(flux) + [0]
            elif mjd_min > epoch_lim[0] and mjd_max < epoch_lim[1]:
                new_mjd = [epoch_lim[0] - mjd_gap] + list(mjd_cent) + [epoch_lim[1] + mjd_gap]
                new_flux = [0] + list(flux) + [0]
    
            # interpolate light curve
            f = interp1d(new_mjd, new_flux, kind='quadratic')
            y_interp = f(epoch_output)
        
            # add line to matrix
            full_matrix.append(y_interp)

    full_matrix = np.array(full_matrix)

    pca = PCA(n_components=npcs)
    pca.fit(full_matrix)

    pcs_output = np.array([[round(item,3) for item in epoch_output]] + list(pca.components_))
    np.savetxt(fname_output, pcs_output, delimiter=',')
    
    
def main(user_input):
    """
    Create principal components from a set of PERFECT simulations.
    
    Parameters
    ----------
    -e: list
        Min and max epoch since maximum brightness to consider.
        Format is [lower_lim, upper_lim].
    -m: str
        Path to header file.
    -n: int
        Number of PCs to extract.
    -p: str
        Path to photo file.
    -o: str
        Output file name.
    -b: float (optional)
        Width of time bin between 2 columns in days. Default is 0.25.
    -d: int (optional)
        extra gap after epoch lim to extrapolate. Avoid boarder issues.
        Default is 5.
    -f: float (optional)
        Minimum flux to avoid noise. Default is 300.
    """
    
    
    fname_header =  user_input.fname_header
    fname_photo = user_input.fname_photo
    npcs = user_input.npcs
    epoch_lim = user_input.epoch_lim
    mjd_gap = user_input.mjd_gap
    flux_lim = user_input.flux_lim
    fname_output = user_input.fname_output
    time_bin = user_input.time_bin
    

    create_PCS(fname_header=fname_header, fname_photo=fname_photo, npcs=npcs,
               epoch_lim=epoch_lim, fname_output=fname_output, 
               mjd_gap=mjd_gap, flux_lim=flux_lim, time_bin=time_bin)
    
    
if __name__ == '__main__':

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='Train random forest for'
                                                 'the KN module')
    parser.add_argument('-e', '--epoch-lim', dest='epoch_lim',
                        required=True, type=int, nargs='+',
                        help='Min and max epoch since maximum brightness.')
    parser.add_argument('-m', '--fname-header', dest='fname_header',
                        required=True, type=str,
                        help='Path to header file.')
    parser.add_argument('-n', '--npcs', dest='npcs', required=True, type=int,
                       help='Number of PCs to extract.')
    parser.add_argument('-p', '--photo-fname', required=True, type=str,
                       help='Photo file name.', dest='fname_photo')
    parser.add_argument('-o', '--fname-output', dest='fname_output', 
                       required=True, type=str, help='Output file name.')
    parser.add_argument('-b', '--time-bin', dest='time_bin', 
                       required=False, default=0.25, type=float,
                       help='Width of time bin.')
    parser.add_argument('-d', '--mjd-gap', dest='mjd_gap', type=int,
                       required=False, default=5, 
                        help='Number of days to extrapolate.')
    parser.add_argument('-f', '--flux-lim', dest='flux_lim', type=float,
                       required=False, default=300, 
                        help='Minimum flux to avoid noise.')
    
    from_user = parser.parse_args()

    main(from_user)