import numpy as np
import dicom
from glob import glob

from skimage.transform import resize
from skimage import morphology, measure
from sklearn.cluster import KMeans


def lung_segmentation(patient_dir):
    """
    Load the dicom files of a patient, build a 3D image of the scan, normalize it to (1mm x 1mm x 1mm) and segment
    the lungs
    :param patient_dir: directory of dcm files
    :return: a numpy array of size (384, 288, 384)
    """

    """ LOAD THE IMAGE """

    # Initialize image and get dcm files
    dcm_list = glob(patient_dir + '/*.dcm')
    img = np.zeros((len(dcm_list), 512, 512), dtype='float32')
    z = []

    # For each dcm file, get the corresponding slice, normalize HU values, and store the Z position of the slice
    for i, f in enumerate(dcm_list):
        dcm = dicom.read_file(f)
        img[i] = float(dcm.RescaleSlope) * dcm.pixel_array.astype('float32') + float(dcm.RescaleIntercept)
        z.append(dcm.ImagePositionPatient[-1])

    # Get spacing and reorder slices
    spacing = map(float, dcm.PixelSpacing) + [np.median(np.diff(np.sort(z)))]
    img = img[np.argsort(z)]

    """ NORMALIZE HU AND RESOLUTION """

    # Clip and normalize
    img = np.clip(img, -1024, 4000)
    img = (img + 1024.) / (4000 + 1024.)

    # Rescale 1mm x 1mm x 1mm
    new_shape = map(lambda x, y: int(x * y), img.shape, spacing[::-1])
    img = resize(img, new_shape, preserve_range=True)

    """ SEGMENT LUNGS USING THRESHOLDING + MORPHOLOGY + SIMPLE RULES """

    # Threshold the image
    middle = img.shape[0] / 2
    data = img[middle].flatten()
    data = data[data > 0][:, None]
    kmeans = KMeans(n_clusters=2).fit(data)
    threshold = np.mean(kmeans.cluster_centers_.flatten())
    thresh_img = np.where(img < threshold, 1.0, 0.0)
    thresh_img[img == 0.] = 0.

    # Clean the image
    thresh_img = morphology.binary_erosion(thresh_img, np.ones([3, 3, 3]))

    # Detect connexity
    labels = measure.label(thresh_img)
    regions = measure.regionprops(labels)
    good_labels = []

    regions = filter(lambda x: x.area > 500000, regions)

    for prop in regions:
        B = prop.bbox
        lim = img.shape[1] / 3
        area_center = np.sum((labels == prop.label)[:, lim:2 * lim, :])

        # Big enough area (1,2,3), not too close to the image border, and with most area in the center
        if B[5] - B[2] > 1 / 4. * img.shape[2] \
                and B[3] - B[0] > 1 / 4. * img.shape[0] \
                and np.sum(B[:3]) > 10 \
                and area_center > 0.3 * prop.area:
            good_labels.append(prop.label)

    lungmask = np.sum([labels == i for i in good_labels], axis=0)

    # Get the entire lung with a big dilation (should use ball(15) but it's too slow)
    for i in range(6):
        lungmask = morphology.binary_dilation(lungmask, np.ones((5, 5, 5)))
    for i in range(4):
        lungmask = morphology.binary_erosion(lungmask, np.ones((5, 5, 5)))

    """ CENTER AND PAD TO GET SHAPE (384, 288, 384) """

    # Center the image

    sum_x = np.sum(lungmask, axis=(0, 1))
    sum_y = np.sum(lungmask, axis=(0, 2))
    sum_z = np.sum(lungmask, axis=(1, 2))

    mx = np.nonzero(sum_x)[0][0]
    Mx = len(sum_x) - np.nonzero(sum_x[::-1])[0][0]
    my = np.nonzero(sum_y)[0][0]
    My = len(sum_y) - np.nonzero(sum_y[::-1])[0][0]
    mz = np.nonzero(sum_z)[0][0]
    Mz = len(sum_z) - np.nonzero(sum_z[::-1])[0][0]

    img = img * lungmask
    img = img[mz:Mz, my:My, mx:Mx]

    # Pad the image to (384, 288, 384)
    nz, nr, nc = img.shape

    pad1 = int((384 - nz) / 2)
    pad2 = 384 - nz - pad1
    pad3 = int((288 - nr) / 2)
    pad4 = 288 - nr - pad3
    pad5 = int((384 - nc) / 2)
    pad6 = 384 - nc - pad5

    # Crop images too big
    if pad1 < 0:
        img = img[:, -pad1:384 - pad2]
        pad1 = pad2 = 0
        if img.shape.shape[0] == 383:
            pad1 = 1

    if pad3 < 0:
        img = img[:, :, -pad3:288 - pad4]
        pad3 = pad4 = 0
        if img.shape.shape[1] == 287:
            pad3 = 1

    if pad5 < 0:
        img = img[:, :, -pad5:384 - pad6]
        pad5 = pad6 = 0
        if img.shape.shape[2] == 383:
            pad5 = 1

    # Pad
    img = np.pad(img, pad_width=((pad1 - 4, pad2 + 4), (pad3, pad4), (pad5, pad6)), mode='constant')
    # The -4 / +4 is here for "historical" reasons, but it can be removed

    return img
