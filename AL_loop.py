import pandas as pd
import numpy as np
import actsnclass
from actsnclass import DataBase


def get_feature_names(npcs=3):
    """
    Create the list of feature names depending on the number of principal components.
    Parameters
    ----------
    npcs : int
        number of principal components to use
    Returns
    -------
    list
        name of the features.
    """
    names_root = ["coeff" + str(i + 1) + "_" for i in range(npcs)] + [
        "residuo_",
        "maxflux_",
    ]

    return [i + j for j in ["g", "r"] for i in names_root]

# this was taken from https://github.com/COINtoolbox/ActSNClass/blob/master/actsnclass/database.py
def build_samples(features: pd.DataFrame, initial_training: int,
                 frac_Ia=0.5, screen=False):
    """Build initial samples for Active Learning loop.
    
    Parameters
    ----------
    features: pd.DataFrame
        Complete feature matrix. Columns are: ['id', 'type', 
        'g_pc_1',  'g_pc_2', 'g_pc_3', 'g_residual', 'g_maxflux',
         'r_pc_1', 'r_pc_2', 'r_pc_3', 'r_residual', 'r_maxflux']
        
    initial_training: int
        Number of objects in the training sample.
    frac_Ia: float (optional)
        Fraction of Ia in training. Default is 0.5.
    screen: bool (optional)
        If True, print intermediary information to screen.
        Default is False.

        
    Returns
    -------
    actsnclass.DataBase
        DataBase for active learning loop
    """
    data = DataBase()
    
    # initialize the temporary label holder
    train_indexes = np.random.choice(np.arange(0, features.shape[0]),
                                     size=initial_training, replace=False)
    
    Ia_flag = features['type'] == 1
    Ia_indx = np.arange(0, features.shape[0])[Ia_flag]
    nonIa_indx =  np.arange(0, features.shape[0])[~Ia_flag]
    
    indx_Ia_choice = np.random.choice(Ia_indx, size=max(1, initial_training // 2),
                                      replace=False)
    indx_nonIa_choice = np.random.choice(nonIa_indx, 
                        size=initial_training - max(1, initial_training // 2),
                        replace=False)
    train_indexes = list(indx_Ia_choice) + list(indx_nonIa_choice)
    
    temp_labels = features['type'][np.array(train_indexes)]

    if screen:
        print('\n temp_labels = ', temp_labels, '\n')

    # set training
    train_flag = np.array([item in train_indexes for item in range(features.shape[0])])
    
    train_Ia_flag = features['type'][train_flag] == 1
    data.train_labels = train_Ia_flag.astype(int)
    data.train_features = features[train_flag].values[:,2:]
    data.train_metadata = features[['id', 'type']][train_flag]
    
    # set test set as all objs apart from those in training
    test_indexes = np.array([i for i in range(features.shape[0])
                             if i not in train_indexes])
    test_ia_flag = features['type'][test_indexes].values == 1
    data.test_labels = test_ia_flag.astype(int)
    data.test_features = features[~train_flag].values[:, 2:]
    data.test_metadata = features[['id', 'type']][~train_flag]
    
    # set metadata names
    data.metadata_names = ['id', 'type']
    
    # set everyone to queryable
    data.queryable_ids = data.test_metadata['id'].values
    
    if screen:
        print('Training set size: ', data.train_metadata.shape[0])
        print('Test set size: ', data.test_metadata.shape[0])
        print('  from which queryable: ', len(data.queryable_ids))
        
    return data


# This was slightly modified from https://github.com/COINtoolbox/ActSNClass/blob/master/actsnclass/learn_loop.py
def learn_loop(data: actsnclass.DataBase, nloops: int, strategy: str,
               output_metrics_file: str, output_queried_file: str,
               classifier='RandomForest', batch=1, screen=True, 
               output_prob_root=None, seed=42, nest=1000, max_depth=None):
    """Perform the active learning loop. All results are saved to file.
    
    Parameters
    ----------
    data: actsnclass.DataBase
        Output from the build_samples function.
    nloops: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    output_metrics_file: str
        Full path to output file to store metric values of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently only 'RandomForest' is implemented.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    n_est: int (optional)
        Number of trees. Default is 1000.
    output_prob_root: str or None (optional)
        If str, root to file name where probabilities without extension!
        Default is None.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    seed: int (optional)
        Random seed.
    max_depth: None or int (optional)
        The maximum depth of the tree. Default is None.
    """

    for loop in range(nloops):

        if screen:
            print('Processing... ', loop)

        # classify
        data.classify(method=classifier, seed=seed, n_est=nest)
        
        if isinstance(output_prob_root, str):
            data_temp = data.test_metadata.copy(deep=True)
            data_temp['prob_Ia'] = data.classprob[:,1]
            data_temp.to_csv(output_prob_root + '_loop_' + str(loop) + '.csv', index=False)
            
        # calculate metrics
        data.evaluate_classification(screen=screen)

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch, seed=seed, screen=screen)
        print('indx: ', indx)
        
        # update training and test samples
        data.update_samples(indx, loop=loop)

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=loop)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop,
                                 full_sample=False)
        


def main():
    
    ###########################################################################
    ############## User input  ################################################
    
    fname_test = 'data/test_features.csv'
    fname_train = 'data/train_features.csv'
       
    initial_training = 10
    strategy = 'UncSampling'
    
    nloops = 1500
    output_metrics_file = 'results/metrics.dat'
    output_queried_file = 'results/queries.dat'
    output_prob_root = None
    
    n_estimators = 30
    max_depth = 42
    
    fname_ini_train = 'results/training_samples/initialtrain.csv'
    fname_queries = 'results/queries.dat'
    fname_fulltrain = 'results/training_samples/fulltrain_depth_42.csv'
    
    remove_zeros = True

    #############################################################################
    
    # read data
    features_names = get_feature_names()
    features_names = ['key', 'y_true'] + features_names
    data_test = pd.read_csv(fname_test)[features_names]
    data_train = pd.read_csv(fname_train)[features_names]
    data_train = data_train.rename(columns={'key':'id', 'y_true':'type'})
    
    if remove_zeros:
        flag = np.array([0.0 not in item for item in data_train.values[:,2:]])
        data_train2 = data_train[flag]
        data_train2.reset_index(inplace=True)
    else:
        data_train2 = data_train
    
    #build samples  
    database = build_samples(data_train2, initial_training=initial_training, screen=True)
    database.features_names = get_feature_names()
    
    # save initial data        
    train = pd.DataFrame(database.train_features, columns=features_names[1:])
    train['objectId'] = database.train_metadata['id'].values
    train['type'] = database.train_metadata['type'].values 
    train.to_csv(fname_ini_train, index=False)
    
    # perform learning loop
    learn_loop(database, nloops=nloops, strategy=strategy, 
               output_metrics_file=output_metrics_file, 
               output_queried_file=output_queried_file,
               classifier='RandomForest', seed=None,
               batch=1, screen=True, output_prob_root=output_prob_root, nest=n_estimators,
               max_depth=max_depth)

    # save final training
    full_train = pd.DataFrame(database.train_features, columns=features_names[2:])
    full_train['objectId'] = database.train_metadata['id']
    full_train['type'] = database.train_metadata['type']
    
    full_train.to_csv(fname_fulltrain, index=False)

if __name__ == '__main__':
    main()