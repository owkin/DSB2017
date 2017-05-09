from glob import glob
from collections import OrderedDict
import numpy as np
from optparse import OptionParser
import os

from lung_segmentation import lung_segmentation
from nodule_segmentation import get_mask
from nodule_features import get_features
from time import time
from keras.models import load_model



def submission(input_dir, output_npz, verbose=False):
    """
    Generates the features for the first model. 3 Unet are applied in the X, Y and Z directions to segment nodules.
    Then handcrafted features (intensity, shape...) are generated from the 10 biggest nodules

    :param input_dir: input directory (subdirectories = patients)
    :param output_npz: name of the npz file where the features will be stored (dictionnary with 6 keys X, Y, Z, I1, I2
    and I3, and values of shape (number of patients, 452) )
    :param verbose: True or False
    """

    patient_list = glob(input_dir + '*')
    patient_list = filter(lambda x: len(x.split('/')[-1]) == 32, patient_list)
    print '{} patients'.format(len(patient_list))
    assert os.listdir(input_dir)

    # Load Unets model (WARNING : in our first code submission, the load was in the script nodule_segmentation.py)
    model = dict((direction, load_model('Unet_{}.hdf5'.format(direction))) for direction in ['X','Y','Z'])

    features = []

    for p in patient_list:

        if verbose:
            print 'Patient {}'.format(p.split('/')[-1])
            t = time()

        try:
            # Lung segmentation
            image = lung_segmentation(p)
            if verbose:
                print 'Scan loaded and lung segmented : {:.2f}s'.format(time() - t)
                t = time()

            # Nodule segmentation with 3 Unet
            masks = OrderedDict()
            for direction in ['X', 'Y', 'Z']:
                masks[direction] = get_mask(image, direction, model[direction])

                if verbose:
                    print 'Unet {} : {:.2f}s'.format(direction, time() - t)
                    t = time()

            # Feature extraction
            features.append(get_features(image, masks))

            if verbose:
                print 'Feature extraction : {:.2f}s'.format(time() - t)
        except:
            print 'Error on patient {}'.format(p)
            patient_list.remove(p)

    radiomics_features = OrderedDict((k, np.array([f[k] for f in features])) for k in ['X', 'Y', 'Z', 'I1', 'I2', 'I3'])
    radiomics_features['ids'] = patient_list
    np.savez(output_npz, radiomics_features)


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option('-i', '--input_dir', metavar='STRING', dest='input_dir', help='input_dir')
    parser.add_option('-o', '--output_npz', metavar='STRING', dest='output_npz', help='output_npz')
    (opts, args) = parser.parse_args()
    input_dir = opts.input_dir
    output_npz = opts.output_npz

    submission(input_dir, output_npz)
