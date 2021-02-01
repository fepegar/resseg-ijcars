from pathlib import Path

from PIL import Image, ImageFont, ImageDraw
from skimage.color import gray2rgb
from skimage import exposure
import scipy.ndimage as ndi
import nibabel as nib
import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


FOREGROUND_INDEX = 1
CHANNELS_DIMENSION_SAMPLE = 0


def plot_segmentation_batch_old(batch, predictions=None):
    with torch.no_grad():
        images = batch['image']['data']
        labels = batch['label']['data']
        mid_slice = int(labels.shape[-1] // 2)
        images = images[..., mid_slice]
        labels = labels[..., mid_slice]
        images_grid = make_grid(
            images,
            normalize=True,
            scale_each=True,
            padding=0,
        )
        images_grid = images_grid.numpy().transpose(1, 2, 0)
        # If the volume has been resampled (e.g. scaling or rotation), a
        # non-linear interpolation (e.g. spline) might have been performed
        # so values will be [0, 1]. We just want the ones after 0.5 for
        # visualization.
        labels = labels > 0.5
        labels_grid = make_grid(labels, padding=0).numpy().transpose(1, 2, 0)
        # Keep only one channel since grayscale
        labels_grid = labels_grid[..., 0]
        images_grid[..., 1][labels_grid > 0.5] = 1  # green
        if predictions is not None:
            predictions = predictions[..., mid_slice]
            foreground = predictions[:, 1:2, ...]
            foreground = foreground > 0.5  # foreground
            predictions_grid = make_grid(
                foreground, padding=0).detach().cpu().numpy().transpose(1, 2, 0)
            # Keep only one channel since grayscale
            predictions_grid = predictions_grid[..., 0]
            images_grid[..., 0][predictions_grid > 0] = 1  # red
            images_grid[..., 2][predictions_grid > 0] = 1  # blue
    fig, axis = plt.subplots()
    axis.imshow(images_grid)
    if 'name' in batch:
        names = [name.split('_t1')[0] for name in batch['name']]
        axis.set_title(' '.join(names))
    plt.tight_layout()
    return fig


def get_regression_grid(batch, predictions) -> np.ndarray:
    with torch.no_grad():
        resected_grid = get_batch_resection_montage(batch, predictions, 'image')
        predictions_grid = get_batch_resection_montage(batch, predictions)

    if len(batch['image']) > 1:
        function = np.vstack
    else:
        function = np.hstack
    grid = function((resected_grid, predictions_grid))
    grid = grid.transpose(2, 0, 1)
    return grid


def get_batch_grid(batch, predictions, regression=False, border=True, transpose=True) -> np.ndarray:
    with torch.no_grad():
        batch_grid = get_batch_overlay_montage(
            batch, predictions, border=border)
        # TODO: get indices beforehand?
        probabilities_grid = get_batch_prediction_montage(predictions, batch)

    if len(batch['image']) > 1:
        function = np.vstack
    else:
        function = np.hstack
    grid = function((batch_grid, probabilities_grid))
    if transpose:
        grid = grid.transpose(2, 0, 1)
    return grid


def plot_segmentation_batch(batch, predictions, border=True):
    grid = get_batch_grid(batch, predictions, border=border)
    fig, axis = plt.subplots()
    axis.imshow(grid)
    if 'name' in batch:
        names = [name.split('_t1')[0] for name in batch['name']]
        axis.set_title(' '.join(names))
    plt.tight_layout()
    return fig


def tensor_to_array(tensor):
    return tensor.squeeze().detach().cpu().numpy()


def normalize_array(array):
    array = array.astype(np.float32)
    array -= array.min()
    if array.max() != 0:
        array /= array.max()
    return array


def reorient_slice(array, a, b):
    """
    a and b are intersection with other planes
    They'll be gray for MRI and white for predictions
    """
    array[a, :] = 0.5
    array[:, b] = 0.5
    return np.fliplr(np.rot90(array))


def rescale_array(array):
    p2, p98 = np.percentile(array, (2, 98))
    rescaled = exposure.rescale_intensity(array, in_range=(p2, p98))
    return rescaled


def get_mid_indices(array):
    dimensions = 3
    shape = size_r, size_a, size_s = array.shape[-dimensions:]
    shape = np.array(shape)
    return shape // 2


def get_slices(array, i, j, k):
    assert array.ndim == 3
    sagittal = reorient_slice(array[i, :, :], j, k)
    coronal = reorient_slice(array[:, j, :], i, k)
    axial = reorient_slice(array[:, :, k], i, j)
    return sagittal, coronal, axial


def get_centroid_slices(label_array):
    foreground = label_array[1] if len(label_array) == 2 else label_array[0]
    coords = np.array(np.where(foreground)).T  # N x 4
    centroid = coords.mean(axis=0).round().astype(np.uint16)
    return centroid


def get_montage(
        array,
        rescale=False,
        normalize=False,
        border=False,
        text=None,
        indices=None,
        ):
    """
    Array should have 3 dimensions?
    """
    assert array.ndim == 3
    if rescale:
        array = rescale_array(array)
    if border:
        # If the volume has been resampled (e.g. scaling or rotation), a
        # non-linear interpolation (e.g. spline) might have been performed
        # so values will be [0, 1]. We just want the ones after 0.5 for
        # visualization.
        array = (array > 0.5).astype(bool)
        eroded = ndi.morphology.binary_erosion(array)
        array = array ^ eroded
    if normalize:
        array = normalize_array(array)

    dimensions = 3
    if indices is None:
        indices = get_mid_indices(array)
    sagittal, coronal, axial = get_slices(array, *indices)
    _, size_a, _ = array.shape[-dimensions:]

    # |  metrics |  axial  |
    # | sagittal | coronal |
    metrics_size = size_a, size_a
    metrics_image = Image.new('F', metrics_size, 0)

    if text is not None:
        add_text(metrics_image, text)

    metrics_array = np.array(metrics_image)
    first_row = np.hstack((metrics_array, axial))
    second_row = np.hstack((sagittal, coronal))
    montage = np.vstack((first_row, second_row))
    return montage


def add_text(image, text):
    font_path = Path('/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf')
    if not font_path.is_file():  # UCL cluster?
        font_path = Path('/usr/share/fonts/dejavu/DejaVuSans.ttf')
    fontsize = 48
    draw = ImageDraw.Draw(image)
    image_size_x, image_size_y = image.size

    font = ImageFont.truetype(str(font_path), fontsize)
    text_size_x, text_size_y = draw.multiline_textsize(text, font=font)
    while text_size_x >= image_size_x:
        fontsize -= 2
        font = ImageFont.truetype(str(font_path), fontsize)
        text_size_x, text_size_y = draw.multiline_textsize(text, font=font)
    start_x = image_size_x // 2 - text_size_x // 2
    start_y = image_size_y // 2 - text_size_y // 2
    xy = start_x, start_y
    gray_value = 200 / 255
    draw.multiline_text(
        xy, text, fill=gray_value, font=font, align='center')


def get_sample_overlay_montage(
        image_tensor,
        label_tensor,
        prediction,
        border=True,
        text=None,
        indices=None,
        ):
    """
    Prediction is [0, 1]
    """
    image = tensor_to_array(image_tensor)
    label = tensor_to_array(label_tensor)

    if label.ndim == 4:
        num_channels = label.shape[CHANNELS_DIMENSION_SAMPLE]
        if num_channels > 1:
            label = label[FOREGROUND_INDEX]
    prediction = prediction[FOREGROUND_INDEX, ...]  # extract foreground
    prediction = tensor_to_array(prediction) >= 0.5
    montage_image = get_montage(
        image, rescale=True, normalize=True, text=text, indices=indices)
    montage_label = get_montage(
        label, border=border, normalize=True, indices=indices)
    montage_prediction = get_montage(
        prediction, border=border, normalize=True, indices=indices)
    montage = gray2rgb(montage_image)
    # If the volume has been resampled (e.g. scaling or rotation), a
    # non-linear interpolation (e.g. spline) might have been performed
    # so values will be [0, 1]. We just want the ones after 0.5 for
    # visualization.
    montage[..., 1][montage_label > 0.5] = 1  # green
    montage[..., 0][montage_prediction > 0] = 1  # red
    montage[..., 2][montage_prediction > 0] = 1  # blue
    return montage


def get_sample_resection_montage(
        image,
        text=None,
        indices=None,
        ):
    """
    Prediction is [0, 1]
    """
    image = tensor_to_array(image)
    montage_image = get_montage(image, rescale=True, normalize=True, text=text, indices=indices)
    montage = gray2rgb(montage_image)
    return montage


def get_resection_indices(batch, i):
    """
    Return voxel index from resection center
    This would not be accurate if spatial transforms are applied
    ESPECIALLY FLIPPING!
    """
    if 'random_resection' not in batch:
        return None
    resection_params = batch['random_resection']
    center_ras = resection_params['resection_center'][i]
    ijk_to_ras = batch['affine'][i]
    ras_to_ijk = np.linalg.inv(ijk_to_ras)
    indices = nib.affines.apply_affine(ras_to_ijk, center_ras)
    indices = indices.astype(np.uint16)
    return indices


def get_batch_resection_montage(batch, predictions, image_key=None, border=True):
    montages = []
    add_names = 'name' in batch

    if add_names:
        for i, prediction in enumerate(predictions):
            image = batch[image_key]['data'][i] if image_key is not None else prediction
            label = batch['label']['data'][i]
            filename = batch['name'][i]
            name = filename.split('_t1')[0]
            indices = get_centroid_slices(label)
            montage = get_sample_resection_montage(image, text=name, indices=indices)
            montages.append(montage)
    else:
        for i, prediction in enumerate(predictions):
            image = batch[image_key]['data'][i] if image_key is not None else prediction
            label = batch['label']['data'][i]
            indices = get_centroid_slices(label)
            montage = get_sample_resection_montage(image, indices=indices)
            montages.append(montage)

    grid = np.hstack(montages)
    return grid


def get_batch_overlay_montage(batch, predictions, border=True):
    montages = []
    add_names = 'name' in batch

    if add_names:
        for i, prediction in enumerate(predictions):
            image = batch['image']['data'][i]
            label = batch['label']['data'][i]
            try:
                filename = batch['name'][i]
                name = filename.split('_t1')[0]
            except Exception:  # getting error when mixing with pseudolabeled
                name = '[Unknown]'
            indices = get_centroid_slices(label)
            montage = get_sample_overlay_montage(
                image, label, prediction, border=border, text=name, indices=indices)
            montages.append(montage)
    else:
        for i, prediction in enumerate(predictions):
            image = batch['image']['data'][i]
            label = batch['label']['data'][i]
            indices = get_centroid_slices(label)
            montage = get_sample_overlay_montage(
                image, label, prediction, border=border, indices=indices)
            montages.append(montage)

    grid = np.hstack(montages)
    return grid


def colorize(prediction):
    # pylint: disable=E1101
    rgba = plt.cm.RdBu_r(prediction)
    rgb = rgba[..., :-1]
    return rgb


def get_batch_prediction_montage(predictions, batch):
    # Predictions are [0, 1]
    montages = []
    for i, prediction in enumerate(predictions):
        label = batch['label']['data'][i]
        indices = get_centroid_slices(label)
        prediction = prediction[FOREGROUND_INDEX, ...]  # extract foreground
        prediction_array = tensor_to_array(prediction)
        montage = get_montage(prediction_array, indices=indices)
        colorized = colorize(montage)
        montages.append(colorized)
    grid = np.hstack(montages)
    return grid
