from optparse import OptionParser
import numpy as np
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import scipy.ndimage
import dicom
import pickle


def getXGBModel(X_train, y_train):
    
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)

    params = {
            "objective": "binary:logistic",   
        
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1,
            "silent": 1,
            "verbose": 0,
            "eval_metric": "logloss",
            "nthread": 32,
            "seed": 338732
    }  
    
    gbm = xgb.train(params, dtrain, 101,  verbose_eval=False)
    return gbm


def get_sje_df(opts):
    sje_features = np.load(opts.input_unet_model)['arr_0'][()]

    cpt = 0
    for i in sorted(sje_features):
        if cpt == 0:
            df_sje = pd.DataFrame(sje_features[i], 
                                  columns=['S' + str(cpt) + '_' + str(j) for j in range(sje_features[i].shape[1])])
        elif i == 'ids':
            df_sje['id'] = [os.path.basename(j) for j in sje_features[i]]
        else:
            df_sje = pd.concat([df_sje, 
                                pd.DataFrame(sje_features[i], columns=['S' + str(cpt) + '_' + str(j) for j in range(sje_features[i].shape[1])])], axis=1)
        cpt += 1

    return df_sje

if __name__ == '__main__':
    
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option('-p', '--input_csv_model_patch', metavar='STRING', dest='input_patch_model', help='input_patch_model')
    parser.add_option('-u', '--input_unet_model', metavar='STRING', dest='input_unet_model', help='input_unet_model')
    parser.add_option('-l', '--labels_csv', metavar='STRING', dest='labels_csv', help='labels_csv')
    parser.add_option('-s', '--save_model', metavar='STRING', dest='save_model', help='save_model')
    
    (opts, args) = parser.parse_args()
    
    model_patch = pd.read_csv(opts.input_patch_model)
    df_sje = get_sje_df(opts)
    labels = pd.read_csv(opts.labels_csv)
    labels.columns = ['id', 'TARGET']

    
    df = pd.merge(model_patch, df_sje, on='id')
    df = pd.merge(df, labels, on='id')

    model = {}
    
    model['xgb_f'] = getXGBModel(df.filter(regex='F').values,  df.TARGET.values).copy()

    for i in range(6):

        X = df.filter(regex='S' + str(i)).values

        model['xgb_s' + str(i)] = getXGBModel(df.filter(regex='S' + str(i)).values,  df.TARGET.values).copy()

    pickle.dump( model, open( opts.save_model, "wb" ))
    
    
    