import numpy as np
from collections import OrderedDict
from skimage import morphology, measure
import radiomics
import SimpleITK as sitk


def get_features(image, masks):
    """
    Extract handcrafted features from the detected nodules using the pyradiomics library

    :param image: input volume of segmented lungs
    :param masks: dictionnary with 3 masks
    :return: a dictionnary with 6 keys and features
    """

    # Features we use from pyradiomics
    feature_keys = [
        'original_shape_Maximum3DDiameter',
        'original_shape_Compactness2',
        'original_shape_Maximum2DDiameterSlice',
        'original_shape_Sphericity',
        'original_shape_Compactness1',
        'original_shape_Elongation',
        'original_shape_SurfaceVolumeRatio',
        'original_shape_Volume',
        'original_shape_SphericalDisproportion',
        'original_shape_Flatness',
        'original_shape_SurfaceArea',
        'original_shape_Maximum2DDiameterColumn',
        'original_shape_Maximum2DDiameterRow',
        'original_firstorder_InterquartileRange',
        'original_firstorder_Skewness',
        'original_firstorder_MeanAbsoluteDeviation',
        'original_firstorder_Energy',
        'original_firstorder_RobustMeanAbsoluteDeviation',
        'original_firstorder_Median',
        'original_firstorder_TotalEnergy',
        'original_firstorder_Maximum',
        'original_firstorder_RootMeanSquared',
        'original_firstorder_90Percentile',
        'original_firstorder_Minimum',
        'original_firstorder_StandardDeviation',
        'original_firstorder_Range',
        'original_firstorder_Variance',
        'original_firstorder_10Percentile',
        'original_firstorder_Kurtosis',
        'original_firstorder_Mean',
        'original_glrlm_ShortRunLowGrayLevelEmphasis',
        'original_glrlm_RunVariance',
        'original_glrlm_GrayLevelNonUniformity',
        'original_glrlm_LongRunEmphasis',
        'original_glrlm_ShortRunHighGrayLevelEmphasis',
        'original_glrlm_RunLengthNonUniformity',
        'original_glrlm_ShortRunEmphasis',
        'original_glrlm_LongRunHighGrayLevelEmphasis',
        'original_glrlm_RunPercentage',
        'original_glrlm_LongRunLowGrayLevelEmphasis',
        'original_glrlm_RunEntropy',
        'original_glrlm_RunLengthNonUniformityNormalized',
    ]

    # We add 3 masks which combine the masks in the X, Y and Z direction
    S = np.sum(masks.values(), axis=0)
    masks['I1'] = (S >= 1).astype('int') # Equivalent to union
    masks['I2'] = (S >= 2).astype('int')
    masks['I3'] = (S == 3).astype('int') # Equivalent to intersection

    # Load images and masks
    features = OrderedDict((k, np.zeros(452, dtype='float')) for k in masks.keys())

    # Features from the N biggest nodules
    N = 10

    for k in masks.keys():

        # Extract regions
        regions = np.array(measure.regionprops(measure.label(masks[k])))
        features[k][-1] = len(regions)
        features[k][-2] = np.sum(masks[k])
        areas = np.array(map(lambda x: x.area, regions))

        order = np.argsort(areas)[::-1]
        regions = regions[order][:N]

        # For each of the min(N, n_nodules) biggest nodules (> 30), extract features
        for i, r in enumerate(regions):
            if r.area < 30:
                break

            nodule_features = OrderedDict()

            # Center
            B = np.array(r.bbox, dtype='float')
            c = (1. / 2 * (B[:3] + B[3:])).astype('int')
            nodule_features['center'] = c / np.array([384., 288., 384.])

            # Extract region of interest (RoI)
            w = 32
            # IMPORTANT : we have a small difference with our initial code here (the 'max').
            # It solves a bug on 3 training patients (99579bbe92d0bb4d5cd98982305870af,
            # a7411bdc623413af46a30bd9d2c41066, da8fea00d3e921e5d8ab4d90a7f6002f.npy)
            # and 1 test patient (1ff375d3f224a510dad6846df1cf19ab)

            X = image[max(0, c[0] - w):c[0] + w, max(0, c[1] - w):c[1] + w, max(0, c[2] - w):c[2] + w]
            Y = masks[k][max(0, c[0] - w):c[0] + w, max(0, c[1] - w):c[1] + w, max(0, c[2] - w):c[2] + w]

            # Extract features
            f = radiomics.featureextractor.RadiomicsFeaturesExtractor().execute(sitk.GetImageFromArray(X),
                                                                                sitk.GetImageFromArray(Y))
            for key in feature_keys:
                nodule_features[key] = f[key]

            if nodule_features['original_shape_Flatness'] > 5.0 or \
                    np.isnan(nodule_features['original_shape_Flatness']):  # bug radiomics
                nodule_features['original_shape_Flatness'] = 5.0

            features[k][i * 45:(i + 1) * 45] = np.hstack(nodule_features.values())

    return features
