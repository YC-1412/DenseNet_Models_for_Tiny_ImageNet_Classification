import os
import zipfile
import urllib.request as url
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# !pip install imgaug
import imgaug as ia
from imgaug import augmenters as iaa


def show_sample(images, images_aug, labels=None):
    """
    Plot the 8 images (index 0~8) and augmented images for visualization and save the figure
    This function is modified from ECBM4040 hw2 image_generator.py https://github.com/ecbme4040/e4040-2020fall-assign2-yc3713/blob/main/utils/image_generator.py
    Args:
        @images: original images to be shown
        @images_aug: augmented images to be shown
        @labels: labels of images, if provided, will show above of orginal images
    """
    fig_path = './figs'
    # If the path doesn't exist, create it first
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    n = 8
    fig, ax = plt.subplots(4,4, figsize=(10,10))
    for i in range(n):
        # plot original image in 1st, 3rd columns
        ax[i%4, i//4*2].imshow(images[i,:])
        # plot augmented image in 2nd, 4th columns
        ax[i%4, i//4*2+1].imshow(images_aug[i,:])
        # trun off axis 
        ax[i%4, i//4*2].axis('off')
        ax[i%4, i//4*2+1].axis('off')
        if labels:
            ax[i%4, i//4*2].set_title(labels[i])
    fig.savefig(fig_path+'/sample_figs.png', bbox_inches='tight')
            
            
def plot_loss(history, model_name):
    '''
    Plot loss curve of a model and save the graph in ./figs
    Args:
        @history: dataframe storing the training and validation loss
        @model_name: str, name of the model
    '''
    fig_path = './figs'
    # If the path doesn't exist, create it first
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(history['epochs'], history['loss'], label='train Loss - {}'.format(model_name))
    ax.plot(history['epochs'], history['val_loss'], label='validation Loss - {}'.format(model_name))
    ax.set_xlabel('Epochs');
    ax.set_ylabel('Loss');
    ax.set_title('Loss per epochs');
    ax.legend();

    fig.savefig(fig_path+'/{}_loss.png'.format(model_name.replace(' ', '_')), bbox_inches='tight')
    

def show_prediction(images, labels, labels_pre, model_name):
    """
    Plot the top 16 images (index 0~15) for visualization and save the graph in ./figs
    This function is modified from ECBM4040 hw2 image_generator.py https://github.com/ecbme4040/e4040-2020fall-assign2-yc3713/blob/main/utils/image_generator.py
    Args:
        @images: images to be shown
    """
    fig_path = './figs'
    # If the path doesn't exist, create it first
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    n = 16
    fig = plt.figure(figsize=(15,15))
    for i in range(n):
        ax = fig.add_subplot(4,4,i+1)
        ax.imshow(images[i,:])
        ax.axis('off')
        ax.set_title('True:'+labels[i]+'\nPred:'+labels_pre[i])
    fig.suptitle('Prediction of {}'.format(model_name),fontsize=20)
    fig.tight_layout()
    fig.savefig(fig_path+'/prediction_{}.png'.format(model_name.replace(' ','_')), bbox_inches='tight')