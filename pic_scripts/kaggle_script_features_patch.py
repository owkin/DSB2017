from optparse import OptionParser
import numpy as np
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from keras.models import Model
from skimage import measure, morphology, segmentation
from keras.utils import np_utils
import cv2
import scipy.ndimage
import dicom
from keras.models import model_from_json
import tensorflow as tf
from scipy.ndimage.interpolation import rotate

def load_scan(path):
    print("Entering load_scan function ...")
    
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    print("Entering get_pixels_hu function ...")
    
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    #image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    image[image <= -1024] = -1024
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    print("Entering resample function ...")
    
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    print("Entering segment_lung_mask function ...")
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image, neighbors=4)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    
    return binary_image

def getSegmentedPatient(patient):
    print("Entering getSegmentedPatient function ...")
    
    x = load_scan(patient)
    first_patient_pixels = get_pixels_hu(x)
    pix_resampled, spacing = resample(first_patient_pixels, x, [1,1,1])
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

    kernel = np.ones((5,5),np.uint8)

    for slicen in range(pix_resampled.shape[0]):
        tmp = segmented_lungs_fill[slicen,:,:].copy()
        tmp = cv2.erode(tmp.astype('int16'),kernel,iterations = 1)
        tmp = cv2.dilate(tmp.astype('int16'),kernel,iterations = 3)
        pix_resampled[slicen,:,:][tmp == 0] = -1024
        
    return pix_resampled

def init_model():
    with tf.device('/gpu:0'):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        session = tf.Session(config=config)
    
        json_file = open('model64x64x64_v5_rotate_v2.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model64x64x64_v5_rotate_v2.h5")
        print("Loaded model from disk")
        
        model2 = Model(input=model.input, output=model.get_layer('flatten_3').output)
        
        return model2
    
def getCube(img_array, model):
    n_size = 64
    
    img_array = img_array.astype('float32')
    img_array += 1024
    img_array[img_array < 0] = 0
    img_array /= 1500
    img_array[img_array > 1] = 1
    
    img_array = rotate(img_array, 90)
    
    zz = []
    zz5 = []
    
    cpt = 0
    for x in range(0, img_array.shape[0]-n_size, n_size/8):
        for y in range(0, img_array.shape[1]-n_size, n_size/8 ):
            for z in range(0, img_array.shape[2]-n_size, n_size/8):
                crop = img_array[x:x+n_size, y:y+n_size, z:z+n_size].copy()
                if np.mean(crop) < 0.05:
                    continue
                zz.append(np.expand_dims(crop, axis=0))
                cpt += 1
                
                
                if cpt % 500 == 0:
                    oo = np.concatenate((zz))
                    print(cpt)
                    if type(oo) == list:
                        continue
                    oo = oo.reshape(oo.shape[0], 64, 64, 64, 1)
                    with tf.device('/gpu:0'):
                        mm = model.predict(oo, batch_size=5)
                        zz5.append(np.expand_dims(np.max(mm, axis=0), axis=0))
                    zz = []
                    
                    
    
    oo = np.concatenate((zz))

    if type(oo) != list:
        oo = oo.reshape(oo.shape[0], 64, 64, 64, 1)
        with tf.device('/gpu:0'):
            mm = model.predict(oo, batch_size=5)
            zz5.append(np.expand_dims(np.max(mm, axis=0), axis=0))
            
    cc = np.concatenate((zz5))
    if type(cc) == list:
        return []
    
    return np.max(cc, axis=0)


if __name__ == '__main__':
    
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option('-i', '--input_dir', metavar='STRING', dest='input_dir', help='input_dir')
    parser.add_option('-o', '--output_csv', metavar='STRING', dest='output_csv', help='output_csv')
    
    (opts, args) = parser.parse_args()
    
    model = init_model()
    z_fin_id = []
    z_fin_feature = []
    
    cpt = 0
    for i in os.listdir(opts.input_dir):
        cpt += 1
        path = opts.input_dir + i + '/'
        print("Processing ID patient", i)
        
        try:
            pix_resampled = getSegmentedPatient(path)
            zz = getCube(pix_resampled, model)
        except:
            continue
            
        z_fin_feature.append(zz)
        z_fin_id.append(i)
        
        
    df = pd.DataFrame(np.vstack(z_fin_feature), columns=['F_' + str(i) for i in range(np.vstack(z_fin_feature).shape[1])])
    df['id'] = z_fin_id
    
    df.to_csv(opts.output_csv, index=False)
