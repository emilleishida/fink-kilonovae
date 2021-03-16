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
import matplotlib.pylab as plt
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


def train_random_forest(features_fname: list, out_model_fname: str, 
                        out_train_fname: str, npcs=int,
                        kn_code = 51, type_key='type',
                        conservative=True, test_size=0.25, n_estimators=1000,
                        external_data=None):
    """Train a Random Forest Classifier and save model to file.
    
    Parameters
    ----------
    features_fname: list
        Path to features files to be used in training.
    out_model_fname: str
        Path to output file where the trained model will be saved.
    out_train_fname: str
        Path to output file where the training sample will be saved.
    npcs: int
        Number of PCs used to extract the features.
    conservative: bool (optional)
        If True add constraint on fit quality. This results in a lower number
        of objects classified as KN but higher purity. Default is True.
    kn_code: list (optional)
        Code identifying the kilonova model. Default is [51].
    n_estimators: int (optional)
        Number of trees in the forest. Default is 1000.
    test_size: float (optional)
        Fraction of data to be used as test sample. Default is 0.25.
    type_key: str (optional)
        Keyword identifying object type in feature matrix. Default is 'type'.
    
    """

    # read features matrix
    dlist = []
    for name in features_fname:
        dtemp = pd.read_csv(name)
        dlist.append(dtemp)
        
    data = pd.concat(dlist, ignore_index=True)

    # consider only objects with at least 1 measurement in each filter        
    if npcs == 1:
        zeros1 = np.logical_or(data['coeff1_g'].values == 0, 
                               data['coeff1_r'].values == 0)
    elif npcs == 2:
        zeros11 = np.logical_or(data['coeff1_g'].values == 0, 
                                data['coeff1_r'].values == 0)
        zeros12 = np.logical_or(data['coeff2_g'].values == 0, 
                                data['coeff2_r'].values == 0)
        zeros1 = np.logical_or(zeros11, zeros12)
    elif npcs == 3:
        zeros11 = np.logical_or(data['coeff1_g'].values == 0, 
                                data['coeff1_r'].values == 0)
        zeros12 = np.logical_or(data['coeff2_g'].values == 0, 
                                data['coeff2_r'].values == 0)
        zeros13 = np.logical_or(data['coeff3_g'].values == 0, 
                                data['coeff3_r'].values == 0)
        zeros1 = np.logical_or(zeros11, np.logical_and(zeros12, zeros13))
    else:
        raise ValueError('Max number of PCs implemented is 3!')

    # constraint on quality cut
    if conservative:
        zeros2 = np.logical_or(data['residuo_g'].values == 0, 
                           data['residuo_r'].values == 0)
        zeros = np.logical_or(zeros1, zeros2)
    else:
        zeros = zeros1
    
    # remove zeros
    data2 = data[~zeros]
    
    # identify KN model
    kn_flag = np.array([item in kn_code for item in data2[type_key].values]) 
    
    # separate data
    X_train, X_test, y_train, y_test = train_test_split(data2,
                                                        kn_flag,
                                                        test_size=test_size)
    
    # build pipeline
    pipe = make_pipeline(RobustScaler(), 
                         RandomForestClassifier(n_estimators=n_estimators))
    
    # fit the Random Forest
    pipe.fit(X_train.values[:,2:], y_train)
    
    # save the model to disk
    pickle.dump(pipe, open(out_model_fname, 'wb'))
    
    # save training sample to disk
    X_train.to_csv(out_train_fname, index=False)
    

def main(user_input):
    """Train and save random forest model to file.
    
    Parameters
    ----------
    -f: list
        Path to features files to be used in training.
    -o: str
        Path to output file where the trained model will be saved.
    -p: int
        Number of PCs used to extract the input feautures. 
        Options are [1, 2, 3].
    -s: str
        Path to output file where the training sample will be save.
    -c: str (optional)
        Keyword identifying object type in feature matrix. Default is 'type'.
    -k: list (optional)
        Code identifying the kilonova model. Default is [51].
    -l: int (optional)
        If True add constraint on fit quality. This results in a lower number
        of objects classified as KN but higher purity. Default is True.
    -n: int (optional)
        Number of trees in the forest. Default is 1000.
    -t: float (optional)
        Fraction of data to be used as test sample. Default is 0.25. 
    """
    
    # get variables
    features_fname = user_input.features_fname
    out_model_fname = user_input.out_model_fname
    out_train_fname = user_input.out_train_fname
    kn_code = user_input.kn_code
    type_key = user_input.type_key
    conservative = bool(user_input.conservative)
    test_size = user_input.test_size
    n_estimators = user_input.n_estimators
    npcs = user_input.npcs
    
    # train and save model
    train_random_forest(features_fname=features_fname, 
                        out_model_fname=out_model_fname, 
                        out_train_fname=out_train_fname,
                        npcs=npcs,
                        kn_code=kn_code, type_key=type_key,
                        conservative=conservative, 
                        test_size=test_size, 
                        n_estimators=n_estimators)
    

if __name__ == '__main__':

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='Train random forest for'
                                                 'the KN module')
    parser.add_argument('-f', '--features-fname', dest='features_fname',
                        required=True, type=str, nargs='+',
                        help='Path to features file.')
    parser.add_argument('-o', '--output-fname', dest='out_model_fname',
                        required=True, type=str,
                        help='Path to output file where the trained'
                        ' model will be saved.')    
    parser.add_argument('-p', '--npcs', dest='npcs', required=True, type=int,
                       help='Number of PCs used to extract the features.')
    parser.add_argument('-s', '--out-train-fname', dest='out_train_fname',
                       required=True, type=str, help='Path to training sample'
                       ' output file name.')
    parser.add_argument('-c', '--model-keyword', dest='type_key',
                        type=str, required=False,
                        default='type',
                        help='Keyword identifying object type in ' 
                        'feature matrix. Default is "type".')
    parser.add_argument('-k', '--kn-code', dest='kn_code',
                        required=False, type=int, default=[51], nargs='+',
                        help='Code identifying the kilonova '
                        'model. Default is [51].')
    parser.add_argument('-l', '--liberal-conservative', 
                         dest='conservative',
                        required=False, default=True, type=int,
                        help='If True add constraint on fit quality. '
                         'Default is True.')
    parser.add_argument('-n', '--n-estimators', dest='n_estimators',
                        required=False, type=int, default=1000,
                        help='Number of trees in the forest.')
    parser.add_argument('-t', '--test-size', dest='test_size',
                        required=False, type=float,
                        help='Fraction of objects to be used in test.'
                        'Default is 0.25.')    
    
    from_user = parser.parse_args()

    main(from_user)