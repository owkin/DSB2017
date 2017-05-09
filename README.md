# Data Science Bowl 2017

In March 2017, we participated to the third [Data Science Bowl challenge organized by Kaggle](https://www.kaggle.com/c/data-science-bowl-2017). This year, the goal was to predict whether a high-riskpatient will be diagnosed with lung cancer within one year, based only on a low-dose CT scan. Our method, which combines deep neural networks and boosted trees, achieved the 10th place on the final leaderboard, over ~2000 teams.

# Method

We developed two methods, both based on nodule detection using the [LUNA dataset](https://luna16.grand-challenge.org/). Both methods pre-process the images to get a fixed 1mm x 1mm x 1mm resolution and segment the lungs using thresholding, morphological operations and connected components selection.

- *Simon’s method* : my method uses 3 U-Net trained in 3 different directions (X, Y and Z) to detect nodules on the LUNA dataset. Given an image of the DSB dataset, I apply these U-Net on all the slices to obtain 6 different 3D binary segmentation masks of the detected nodules (directions X, Y, Z + unions I1, I2 and I3). From these masks, I extract the 10 biggest nodules and compute 45 features for each nodule (including location, area, volume, HU statistics etc.). I then use these features 450 (+2) features as an entry to an XGboost classifier  and average the 6 predictions. This pipeline takes around 2 to 3 minutes per patient. 

 Each U-net takes a stack of 11 slices as input and outputs a nodule segmentation mask of the 6th slice. It permits to take into account 3D information and improves the dice score from 0.50 for a single input (6th slice) to 0.75 for the 3D stack. 

- *Pierre’s method* : my method is a cube-based method using 3D convolutional network. As you may notice, this method is quite similar to the r4m0n approach. 
1. Extract 64x64x64 chunks from all entries on the LUNA16 annotations.csv file, and a random sample of the non-nodule entries from candidates.csv as a negative reference. The  LUNA16 annotations_excluded.csv file gave me worse result.
2. Train a 3D VGG derivative on augmented entries (especially flip and translation) from the features above with the labels being classes based on the size of the nodules
3. Chop up each scan from the stage 1 dataset into overlapping chunks to fit the above network, capturing the final convolution step (before FC layer)
4. Build a feature set from the output above, aggregating over all chunks 
5. Train a XGB classifier with the features and the stage 1 labels
6. Evaluate the classifier on the test set
 It was hard to find a good network architecture, especially because a good performance on the Luna16 dataset doesn’t necessarily mean a good performance on the kaggle dataset.

To blend our two methods we simply average the predictions.

# Requirements

This code requires the following packages to be installed :
- Tensorflow-gpu 1.0
- [xgboost](https://xgboost.readthedocs.io/en/latest/)
- Keras 2.0
- openCV
- pyradiomics

 To install pyradiomics please refer to http://pyradiomics.readthedocs.io/en/latest/installation.html.
 If you have the error ImportError: cannot import name _cmatrices try $sudo python setup.py develop

# Run the model

To run the script, you will have to run the 6 commands below. The 2 first generate the train features from the Pierre's and Simon's model, the 2 following generate the test features, the 5th train the XGboost model on the train features, and the 6th apply it on the test and save the csv submission file. Note that we parallelized this code for the competition. **Please replace XXX by your data path*

```
#Create train features with patch model (Pierre's model)
 cd ./pic_scripts/
python kaggle_script_features_patch.py -i /XXX/train/ -o /tmp/patch_model_features_train.csv

#Create train features with unet+radiomics (Simon's model)
cd ./../sje_scripts/
python kaggle_script_features_unet.py -i /XXX/train/ -o /tmp/unet_features_train.npz

#Create test features with patch model (Pierre's model)
cd ./../pic_scripts/
python kaggle_script_features_patch.py -i /XXX/test/ -o /tmp/patch_model_features_test.csv 

#Create test features with unet+radiomics (Simon's model)
cd ./../sje_scripts/
python kaggle_script_features_unet.py -i /XXX/test/ -o /tmp/unet_features_test.npz

#Build xgbmodel from train features
cd ./../pic_scripts/
python kaggle_train.py -p /tmp/patch_model_features_train.csv -u /tmp/unet_features_train.npz -l /fusionio/KaggleBowl/stage1_labels.csv -s /tmp/model.bst

#Prediction from test features
python kaggle_predict.py -p /tmp/patch_model_features_test.csv -u /tmp/unet_features_test.npz -l /fusionio/KaggleBowl/sample_submission.csv -s /tmp/model.bst -f /tmp/sub_final.csv```
