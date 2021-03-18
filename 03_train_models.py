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
                        out_train_fname: str, out_test_fname: str, npcs=int,
                        kn_code = [51], type_key='type',
                        test_size=0.5, n_estimators=30,
                        external_data=None):
    """Train a Random Forest Classifier and save model to file.
    
    Parameters
    ----------
    features_fname: list
        Path to features files to be used in training.
    out_model_fname: str
        Path to output file where the trained model will be saved.
    out_test_fname: str
        Path to output file where the test sample will be saved.
    out_train_fname: str
        Path to output file where the training sample will be saved.
    npcs: int
        Number of PCs used to extract the features.
    kn_code: list (optional)
        Code identifying the kilonova model. Default is [51].
        If kn_code = [None], train the model with multiple classes.
    n_estimators: int (optional)
        Number of trees in the forest. Default is 30.
    test_size: float (optional)
        Fraction of data to be used as test sample. Default is 0.5.
    type_key: str (optional)
        Keyword identifying object type in feature matrix. Default is 'type'.
    
    """

    # read features matrix
    dlist = []
    for name in features_fname:
        dtemp = pd.read_csv(name)
        dlist.append(dtemp)
        
    data_orig = pd.concat(dlist, ignore_index=True)
    
    # remove bad residuo fits
    bad_fit_flag = np.logical_or(data_orig['residuo_g'].values == 0,
                                  data_orig['residuo_r'].values == 0)
    
    data = data_orig[~bad_fit_flag]
    
    # identify KN model
    if str(kn_code[0]) != 'None':
        kn_flag = np.array([item in kn_code for item in data[type_key].values]) 
    else:
        kn_flag = data[type_key].values.astype(int)

    # separate data
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        kn_flag,
                                                        test_size=test_size)
    
    # build pipeline
    pipe = make_pipeline(RobustScaler(), 
                         RandomForestClassifier(n_estimators=n_estimators))
    
    # fit the Random Forest
    pipe.fit(X_train.values[:,2:], y_train)    
    
    # save the model to disk
    pickle.dump(pipe, open(out_model_fname, 'wb'))
    
    # save samples to disk
    X_train.to_csv(out_train_fname, index=False)
    X_test.to_csv(out_test_fname, index=False)
    

def main(user_input):
    """Train and save random forest model to file.
    
    Parameters
    ----------
    -e: str
        Path to output file where the test sample will be saved.
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
        Code identifying the kilonova model. Default is [51,50].
    -n: int (optional)
        Number of trees in the forest. Default is 30.
    -t: float (optional)
        Fraction of data to be used as test sample. Default is 0.5. 
    """
    
    # get variables
    features_fname = user_input.features_fname
    out_model_fname = user_input.out_model_fname
    out_train_fname = user_input.out_train_fname
    out_test_fname = user_input.out_test_fname
    kn_code = user_input.kn_code
    type_key = user_input.type_key
    test_size = user_input.test_size
    n_estimators = user_input.n_estimators
    npcs = user_input.npcs

    # train and save model
    train_random_forest(features_fname=features_fname, 
                        out_model_fname=out_model_fname, 
                        out_train_fname=out_train_fname,
                        out_test_fname=out_test_fname,
                        npcs=npcs,
                        kn_code=kn_code, type_key=type_key,
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
                       ' output file.')
    parser.add_argument('-e', '--out-test-fname', required=True, type=str,
                       help='Path to test sample output file.')
    parser.add_argument('-c', '--model-keyword', dest='type_key',
                        type=str, required=False,
                        default='type',
                        help='Keyword identifying object type in ' 
                        'feature matrix. Default is "type".')
    parser.add_argument('-k', '--kn-code', dest='kn_code',
                        required=False, default=[51,50], nargs='+',
                        help='Code identifying the kilonova '
                        'model. Default is [51].')
    parser.add_argument('-n', '--n-estimators', dest='n_estimators',
                        required=False, type=int, default=30,
                        help='Number of trees in the forest.'
                       'Default is 30.')
    parser.add_argument('-t', '--test-size', dest='test_size',
                        required=False, type=float, default=0.5,
                        help='Fraction of objects to be used in test.'
                        'Default is 0.5.')    
    
    from_user = parser.parse_args()

    main(from_user)