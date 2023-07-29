import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=18)

import czifile

import sys
sys.path.append('/home/chase/codes/python_functions')
import plotting as my_plot


def get_aspect_ratio(file, scan='xz'):
    """Extracts the voxel dimensions from metadata and computes the xz aspect ratio"""
    f = czifile.CziFile(file)
    m = f.metadata(raw=False)['ImageDocument']['Metadata']['Scaling']['Items']
    x = m['Distance'][0]['Value']
    y = m['Distance'][1]['Value']
    z = m['Distance'][2]['Value']
    assert x - y <= 1e-14
    if scan == 'xz' or scan == 'z':
        aspect = z/x
        return aspect, x
    elif scan == 'xy' or scan == 'y':
        aspect = y/x
        return aspect, x

def read_image(file, gamma=1, epsilon=0, scan='xz'):
    image = czifile.imread(file)

    if scan == 'xz' or scan == 'z':
        data = image[0, 0, :, 0, :, 0, :, 0]
        shape = np.shape(data)
        new = np.zeros((shape[1], shape[2], 3))
        # Normalization for 12-bit intensity
        for i in range(3):
            new[:, :, i] = (image[0, 0, i, 0, :, 0, :, 0]/(2**12 - 1))

    elif scan == 'xy' or scan == 'y':
        data = image[0, 0, :, 0, 0, :, :, 0]
        shape = np.shape(data)
        new = np.zeros((shape[1], shape[2], 3))
        # Normalization for 12-bit intensity
        for i in range(3):
            new[:, :, i] = (image[0, 0, i, 0, 0, :, :, 0]/(2**12 - 1))

    # Manual implementation of power-law colormap normalization with a lower threshold (epsilon)
    # Default parameters are for linear scaling
    alpha = 1.0 / ((1.0 - epsilon)**gamma)
    new[new < epsilon] = epsilon
    new = alpha * (new - epsilon)**gamma
    return new

def add_scale_bar(ax, x):
    scalebar = AnchoredSizeBar(ax.transData,
                           int(25e-6/x), r'25 $\mu$m', 'lower center',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fontprops)
    _ = ax.add_artist(scalebar)
    return

axis_map = {0:1, 1:0, 2:2}

def plot_color_montage(aspect, x, new, n_colors=3, scale_bar=True):
    """Order:  composite, large, small, mAb"""
    shape = np.shape(new)

    n = 3 # scales the image size in jupyter - ultimately only matters relative to the scale bar text size
    fig, ax = plt.subplots(n_colors+1, 1, figsize=(n*aspect, n*4))
    plt.subplots_adjust(hspace=0)

    ax[0].imshow(new, aspect=aspect)
    ax[0].axis('off')
    if scale_bar:
        add_scale_bar(ax[0], x)

    for i in range(n_colors):
        temp = np.zeros(shape)
        temp[:,:,i] = new[:,:,i]
        ax[axis_map[i]+1].imshow(temp, aspect=aspect)
        ax[axis_map[i]+1].axis('off')
        if scale_bar:
            add_scale_bar(ax[i+1], x)

    return fig, ax

def plot_greyscale_montage(aspect, x, new):
    """Order:  large, small, mAb"""
    n = 3 # scales the image size in jupyter - ultimately only matters relative to the scale bar text size
    fig, ax = plt.subplots(3, 1, figsize=(n*aspect, n*3))
    plt.subplots_adjust(hspace=0)

    for i in range(3):
        ax[axis_map[i]].imshow(new[:,:,i], aspect=aspect, cmap='gray') # jet, hot
        ax[axis_map[i]].axis('off')
        add_scale_bar(ax[i], x)

    return fig, ax
