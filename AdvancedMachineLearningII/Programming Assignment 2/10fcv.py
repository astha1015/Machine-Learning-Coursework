import os
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

df =  pd.read_csv('iris.data.txt', header = None, delimiter='\t')

features = df.iloc[:,0:-1]
y = df.iloc[:,-1]

y = pd.DataFrame(y)

def create_10Folds(source_path, destination_path):
  df =  pd.read_csv(source_path)
 
  
  features = df.iloc[:,0:-1]
  y = df.iloc[:,-1]

  np_features = np.array(features.values)
  np_area = np.array(y.values)
  kfold = KFold(n_splits=10)
  fold = 0

  for train, test in kfold.split(features, y):
    
    fold+=1
    
    train_output = destination_path + 'train' + str(fold) + '.txt'
    test_output = destination_path + 'test' + str(fold) + '.txt'
    
    feature_fold_train = np_features[train]
    area_fold_train = np_area[train]
  
    feature_fold_test = np_features[test]
    area_fold_test = np_area[test]
  
    feature_fold_train_frame = pd.DataFrame(feature_fold_train)
    feature_fold_test_frame = pd.DataFrame(feature_fold_test)
  
    area_fold_train_frame = pd.DataFrame(area_fold_train)
    area_fold_test_frame = pd.DataFrame(area_fold_test)
  
    result_train = pd.concat([feature_fold_train_frame, area_fold_train_frame], axis = 1)
    result_test = pd.concat([feature_fold_test_frame, area_fold_test_frame], axis = 1)
  
    result_train = pd.DataFrame(result_train)
    result_train.to_csv(train_output, index=False, header=False)
  
    result_test = pd.DataFrame(result_test)
    result_test.to_csv(test_output, index=False, header=False)

x_update = features 
y_update = y 
x_sparse = coo_matrix(x_update)
x_update, x_sparse, y_update = shuffle(x_update, x_sparse, y_update, random_state=0)

  
dataframe = pd.concat([x_update,y_update], axis = 1)

np.savetxt("Processed.irisdata.txt", dataframe, fmt='%.1f, %.1f, %.1f, %.1f, %03d' )
create_10Folds('Processed.irisdata.txt', 'folds/')
