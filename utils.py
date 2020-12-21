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

def download_data(path):
    """
    Download the Tiny-ImageNet data from the website, which is approximately 240MB.
    The data (a .tar.gz file) will be store in the given path.
    Args:
        @path: path where the data saved/to be saved
    """
    # If the path doesn't exist, create it first
    if not os.path.exists(path):
        os.mkdir(path)
    # If the data hasn't been downloaded yet, download it
    if not os.path.exists(path+'/tiny-imagenet-200.zip'):
        print('Start downloading data...')
        url.urlretrieve("http://cs231n.stanford.edu/tiny-imagenet-200.zip",
                        path+"/tiny-imagenet-200.zip")
        print('Download complete.')
    else:
        print('Tiny-ImageNet package already exists.')
        
def unpack_data(path):
    """
    Unpack the tiny-imagenet-200.zip. The unpacked data will be store in the given path.
    Args:
        @path: path where the data saved/to be saved
    """
    # If the data hasn't been downloaded yet, download it first
    if not os.path.exists(path+'/tiny-imagenet-200.zip'):
        download_data(path)
        
    else:
        print(path+'/tiny-imagenet-200.zip already exists')
    # Check if the package has been unpacked, otherwise unpack the package
    if not os.path.exists(path+'/tiny-imagenet-200/'):
        print('Begin extracting...')
        with zipfile.ZipFile(path+'/tiny-imagenet-200.zip', 'r') as zip_ref:
            zip_ref.extractall(path)
        print('Unzip complete.')
    else:
        print(path+'/tiny-imagenet-200 already unzipped.')
        
             
def load_data(path, img_width=64, img_height=64, batch_size=128, augmentation=None, seed=None, load_test=False):
    '''
    Create generators of rescaled train and validation data with required image size from Tiny ImageNet dataset
    Training data is stored in different folders, with each folder represents a class
    Training data is split into training and validation data
    Images in "val" folder are used as testing data. They are stored in one folder, with a text file storing the image file names and corresponding classes
    
    Dataset can be downloaded here: http://cs231n.stanford.edu/tiny-imagenet-200.zip
    Data loading functions (flow_from_directory, flow_from_dataframe) inspired by https://keras.io/api/preprocessing/image/
    This function is modified from ECBM4040 hw2 task5 kaggle.py https://github.com/ecbme4040/e4040-2020fall-assign2-yc3713/blob/main/ecbm4040/neuralnets/kaggle.py
    
    Args:
        @train_path: directory where the training data is stored
        @val_path: folder where the validation data is stored
        @img_width: image width of loaded data
        @img_height: image height of loaded data
        @batch_size: batch size of the data
        @augmentation: preprocessing function of data augmentation, default: None
    
    Return:
        train_generator: generator of training data
        val_generator: generator of validation data
        test_generator: generator of testing data
    '''
    # If the data does not exist, unpack the zip file or download it first.
    if not os.path.exists(path+'/tiny-imagenet-200/'):
        unpack_data(path)
    else:
        print('Begin loading...')
    
    train_path = path+'/tiny-imagenet-200/train'
    val_path = path+'/tiny-imagenet-200/val'
    
    # load and rescale training data
    # if augmentation=None, no data augmentation
    # split 20% training data as validation set 
    train_datagen = ImageDataGenerator(preprocessing_function=augmentation, rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=(img_width, img_height),
                                                        class_mode='categorical',   # classification problem
                                                        batch_size=batch_size,
                                                        subset='training',
                                                        shuffle=True, seed=seed)     # shuffle the data
    
    val_generator = train_datagen.flow_from_directory(train_path,
                                                      target_size=(img_width, img_height),
                                                      class_mode='categorical',   # classification problem
                                                      batch_size=batch_size,
                                                      subset='validation',
                                                      shuffle=True, seed=seed)     # shuffle the data
    print('Training data shape: {}'.format((train_generator.n,)+train_generator.next()[0].shape[1:]))
    print('Validation data shape: {}'.format((val_generator.n,)+val_generator.next()[0].shape[1:]))
    
    if not load_test:
        # don't load test data
        train_generator.reset()
        val_generator.reset()
        print('End loading!')
        return train_generator, val_generator
    else:
        # load test data
        # the images in "val" folder are used as test data
        # val_annotations.txt fields: 
        # filename, class, x-axis of the center of the bounding box, y-axis of the center, width of the box, height 
        # use only the filename, class fields for this project
        test_df = pd.read_csv(val_path+'/val_annotations.txt', 
                              sep='\t', header=None, names=['filename','class','x','y','width','height'], 
                              usecols=['filename','class'])

        # load and rescale validation data
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_dataframe(test_df,
                                                          directory=val_path+'/images/',
                                                          x_col=test_df.columns[0],
                                                          y_col=test_df.columns[1],
                                                          target_size=(img_width, img_height),
                                                          class_mode='categorical',
                                                          batch_size=batch_size,
                                                          shuffle=True, seed=seed)
        print('Testing data shape: {}'.format((test_generator.n,)+test_generator.next()[0].shape[1:]))

        train_generator.reset()
        val_generator.reset()
        test_generator.reset()
        print('End loading!')
        return train_generator, val_generator, test_generator




def data_augmentation(complicated=False,
                      CoarseDropout_range=(0.0, 0.05),
                      CoarseDropout_size_percent=(0.02, 0.25),
                      Affine_translate_percent=(-0.2, 0.2),
                      Affine_scale=(0.5, 1.5),
                      Affine_shear=(-20, 20),
                      Affine_rotate=(-45, 45),
                      Flip_percent=0.5,
                      GaussianBlur_sigma=(0.0, 3.0),
                      CropAndPad_percent=(-0.25, 0.25),
                      Multiply=(0.5, 1.5),
                      LinearContrast=(0.4, 1.6),
                      AdditiveGaussianNoise_scale=0.2*255):
    '''
    Generate a sequence of data augmentation steps such as rotate, flip, scale, etc. 
    A random number of augmentation steps is implemented
    Augmentation steps are similar to that in the original paper
    Details of aumnetation methods and functions can be found here: https://github.com/aleju/imgaug
    
    Args:
        @complicated: boolean value indicating whether complicated augmentation is applied. True: augmentation for network 2, false: augmentation for network 1
        @CoarseDropout_range(x1,x2): Drop x1 to x2% pixels by converting them to black pixels
        @CoarseDropout_size_percent(x1,x2): do CoarseDropout on an image that has x1 to x2% of the original size
        @Affine_translate_percent(x1,x2): Translate images by x1 to x2% on x- and y-axis
        @Affine_scale(x1,x2): Scale images to a value of x1 to x2% of their original size on x- and y-axis
        @Affine_shear(x1,x2): Shear images by x1 to x2 degrees on x- and y-axis
        @Affine_rotate(x1,x2): Rotate images by x1 to x2 degrees 
        @Flip_percent: Flip x% of all images
        @GaussianBlur_sigma: Blur each image with a gaussian kernel with a sigma of x
        @CropAndPad_percent: Crop or pad each side by up to x% relative to its original size
        @Multiply(x1,x2): Multiply each image with a random value between x1 and x2
        @LinearContrast: Modify the contrast of images according to 127 + alpha*(v-127)`, where v is a pixel value and alpha is sampled uniformly from the interval [x1, x2]
        @AdditiveGaussianNoise_scale: Add gaussian noise to an image, sampled once per pixel from a normal distribution N(0, s), where s is sampled per image and varies between x1 and x2
    
    Return:
        aug: augmentation function
        aug_images: augmented images
    '''
    ia.seed(123)
    
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    if complicated:
        # augmentation for network 2
        aug = iaa.Sequential(
                         
                         [
                             # apply to all images
                             iaa.Fliplr(Flip_percent),   # Flip/mirror horizontally
                             iaa.Flipud(Flip_percent),   # Flip/mirror vertically
                             
                             # apply to half of images
                             # sets rectangular areas within images to zero
                             sometimes(iaa.CoarseDropout(CoarseDropout_range, 
                                                         size_percent=CoarseDropout_size_percent)), 
                             # Apply affine transformations to images
                             sometimes(iaa.Affine(translate_percent=Affine_translate_percent,
                                                  scale=Affine_scale,
                                                  shear=Affine_shear,
                                                  rotate=Affine_rotate
                                                 )),  
                             # blur images using gaussian kernels
                             sometimes(iaa.GaussianBlur(sigma=GaussianBlur_sigma)),
                             # crop and pad images, and resize images back to their original size
                             sometimes(iaa.CropAndPad(percent=CropAndPad_percent)), 
                             # Multiply all pixels with a specific value
                             # thereby making the image darker or brighter
                             sometimes(iaa.Multiply(Multiply)), 
                             # 127 + alpha*(v-127)`, where v is a pixel value
                             # alpha is sampled uniformly from the interval
                             sometimes(iaa.LinearContrast(LinearContrast)) 
                         ], random_order=True)
    else:
        
        # augmentation for network 1
        aug = iaa.SomeOf((0, None),
                         [
                             # sets rectangular areas within images to zero
                             iaa.CoarseDropout(CoarseDropout_range, 
                                               size_percent=CoarseDropout_size_percent), 
                             # Apply affine rotation on the y-axis to input data
                             iaa.Affine(scale=Affine_scale,
                                        rotate=Affine_rotate
                                       ),  
                             # crop and pad images, and resize images back to their original size
                             iaa.CropAndPad(percent=CropAndPad_percent, keep_size=True),  
                             # Add gaussian noise to an image
                             iaa.AdditiveGaussianNoise(scale=AdditiveGaussianNoise_scale) 
                         ], random_order=True)
    return aug.augment_image

def get_lookup_tables(path, generator):
    '''
    Generate lookup table of correct labels of images
    Args:
        @path: path where the loopup table [class_id, class_name] is
        @generator: generator of the data that need to lookup
    Return:
        lookup: dictionary {index: class_name}
    '''
    # generate [class_id, class_name] lookup table
    lookup_des = pd.read_csv(path+'/tiny-imagenet-200/words.txt', 
                           sep='\t', header=None, names=['class_id','class_name'])
    lookup_des['class_name'] = lookup_des['class_name'].str.split(',').str[0]
    # generate [index, class_id] lookup table. Index is the index in train_generator
    lookup_id = pd.DataFrame(([y,x] for x,y in generator.class_indices.items()),columns=['indexes','class_id'])
    # generate {index: class_name} lookup dict
    lookup = lookup_des.merge(lookup_id, on='class_id', how='right').set_index('indexes')['class_name'].to_dict()
    return lookup

def get_labels(path, indexes, generator):
    '''
    Get labels of images given the image class index from generator
    Args:
        @indexes: boolean array of size 200, showing index of image class from generator
        @generator: spcify training or validation generator
    Return:
        labels: labels of images
    Call: get_lookup_tables(path, generator)
    '''
    lookup = get_lookup_tables(path, generator)
    labels = [lookup[i] for i in np.where(indexes == 1)[1]]
    return labels


            
def load_history(checkpoint_filepath, history_ls):
    '''
    load checkpoint history of the model. History stores accuracy, loss, lr, val_accuracy, val_loss
    Args:
        @checkpoint_filepath: path where history log files are stored
        @history_ls: list of names of history log files
    Return:
        history: dataframe of history
    '''
    if len(history_ls) == 0:
        # if no log file
        return(print('No log file is founded'))
    
    history = pd.read_csv(checkpoint_filepath+'/'+history_ls[0], sep=',')
    if len(history_ls) > 1:
        # concate all log files into one dataframe
        for name in history_ls[1:]:
            history = pd.concat([history, pd.read_csv(checkpoint_filepath+'/'+name, sep=',')])
    # renumber epochs
    history.reset_index(drop=True,inplace=True)
    history['epochs'] = history.index.to_list()
    return history

