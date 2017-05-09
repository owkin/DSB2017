from optparse import OptionParser
import numpy as np
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import scipy.ndimage
import dicom
import pickle


def getXGBPrediction(X_test, gbm):
    
    dtest = xgb.DMatrix(X_test, missing=np.nan)
    return gbm.predict(dtest)


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
    parser.add_option('-s', '--load_model', metavar='STRING', dest='load_model', help='load_model')
    parser.add_option('-f', '--submission_file', metavar='STRING', dest='submission_file', help='submission_file')
    
    (opts, args) = parser.parse_args()
    
    df_model_patch = pd.read_csv(opts.input_patch_model)
    df_sje = get_sje_df(opts)
    labels = pd.read_csv(opts.labels_csv)
    labels.columns = ['id', 'TARGET']

    
    df = pd.merge(df_model_patch, df_sje, on='id')
    df = pd.merge(df, labels, on='id')

    model = pickle.load( open( opts.load_model, "rb" ) )
    
    res_f = getXGBPrediction(df.filter(regex='F'), model['xgb_f'])
    
    res_s = getXGBPrediction(df.filter(regex='S'), model['xgb_s0'])
    for i in range(1, 6, 1):
        res_s += getXGBPrediction(df.filter(regex='S'), model['xgb_s' + str(i)])
        
    res_s /= 6.0
    
    #We made a slight change here
    #We finally decide to apply an uniform weighting of our 2 models (instead of 1/3 and 2/3)  
    #res_f is pierre's model predictions and res_s is simon's model predictions
    df['cancer'] = (res_f+res_s)/2.0
    
    df = pd.merge(labels, df, on='id', how='left').fillna(0.2)
    df[['id', 'cancer']].to_csv(opts.submission_file, index=False)
    
    
    
    
    
    