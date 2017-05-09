import numpy as np
import keras.backend as K
import keras

SMOOTH = 0.001


# Loss used to train the models (needed to load a keras model)
def real_dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
    summation = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3))
    return K.mean((2. * intersection + SMOOTH) / (summation + SMOOTH))


def real_dice_coef_loss(y_true, y_pred):
    return - real_dice_coef(y_true, y_pred)


keras.losses.real_dice_coef_loss = real_dice_coef_loss


def get_mask(img, direction, model):
    """
    For an input image of shape (384, 288, 384), apply a U-Net to each slice in direction X, Y or Z to output a mask
    with the same shape where the 1s represent the nodules detected by the Unet
    """

    # Load model
    assert direction in ['X', 'Y', 'Z']

    # Parameters
    n_slices = 11  # The input of the U Net is not a single image but a volume of n_slices images
    smin = n_slices / 2
    smax = n_slices - smin
    axis1 = {'X': (0, 1), 'Y': (0, 2), 'Z': (1, 2)}[direction]
    axis2 = {'X': 2, 'Y': 1, 'Z': 0}[direction]  # axis along we crop and select slices

    # We only extract masks on non empty slices located in the variable position
    sum_axis = np.sum(img, axis=axis1)
    inf = np.nonzero(sum_axis)[0][0]
    sup = img.shape[axis2] - np.nonzero(sum_axis[::-1])[0][0]
    inf = max(inf, smin)
    sup = min(sup, img.shape[axis2] - smax + 1)
    positions = np.arange(inf, sup, 1)
    mask = np.zeros(img.shape)

    for j, i in enumerate(positions):
        inputs = np.take(img, range(i - smin, i + smax), axis=axis2)
        inputs = np.rollaxis(inputs, axis2, 0)[None, None, :, :, :]
        if direction == 'X':
            mask[:, :, i] = model.predict(inputs)[0, 0]
        elif direction == 'Y':
            mask[:, i, :] = model.predict(inputs)[0, 0]
        elif direction == 'Z':
            mask[i, :, :] = model.predict(inputs)[0, 0]

    mask = (mask > 0.5).astype('int16')
    return mask
