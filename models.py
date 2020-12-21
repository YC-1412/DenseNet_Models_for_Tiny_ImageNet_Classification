import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.nn import space_to_depth
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.regularizers import l2
from utils import get_lookup_tables, get_labels
from visualization import show_prediction

def init_model_1(input_shape):
    '''
    Build model 1 with the same struction as described in the original paper.
    For model 1, two blocks are concatenated after the maxpooling layer.
    
    Args:
        @input_shape: shape of input images. For this dataset, it should be (None, None, 3)
    Returns:
        @model: CNN model
    Call: 
        add_block(inputs, filter_n=[128], activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=None)
        concate_blocks(block1, block2)
    '''
    # init input layer
    inputs = Input(shape=input_shape)
    
    # block 1
    # MaxPooling layer at the end of each block
    # add a block of 5 layers, inside [] is the size of each layer
    block1 = add_block(inputs, [32, 64, 128, 256, 512], use_bias=False)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2)(block1)

    # block 2
    # MaxPooling layer at the end of each block
    block2 = add_block(block1, [64, 128, 256, 512, 1024], use_bias=False)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2)(block2)
    
    # concat the result of two blocks as the input of next layer
    block2 = concate_blocks(block1, block2)

    # block 3
    # MaxPooling layer at the end of each block
    block3 = add_block(block2, [32, 128, 256, 512, 1024], use_bias=False)
    block3 = MaxPooling2D(pool_size=(2, 2), strides=2)(block3)

    block3 = concate_blocks(block2, block3)
    
    conv = Conv2D(200, (1,1), strides=(1,1), use_bias=False)(block3)
    bn = BatchNormalization()(conv)
    avgpool = GlobalAveragePooling2D()(bn)

    output = Activation('softmax')(avgpool)

    model = Model(inputs=inputs, outputs=output)

    return model

def init_model_2(input_shape):
    '''
    Build model 2 with the same struction as described in the original paper.
    For model 2, two blocks are concatenated before the maxpooling layer.
    
    Args:
        @input_shape: shape of input images. For this dataset, it should be (None, None, 3)
    Returns:
        @model: CNN model
    '''
    kernel_initializer = 'VarianceScaling'
    kernel_regularizer = l2(2e-4)
    
    inputs = Input(shape=input_shape)
    layer1 = add_layer(inputs, 32, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    
    # block 1
    # add a block of 4 layers, inside [] is the size of each layer
    block1 = add_block(layer1, [128]*4, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    # concat the result of two blocks as the input of next layer
    block1 = Concatenate()([layer1, block1])
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2)(block1)
    
    # block 2
    block2 = add_block(block1, [256]*4, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    block2 = Concatenate()([block1, block2])
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2)(block2)
    
    # block 3
    block3 = add_block(block2, [512]*4, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    block3 = Concatenate()([block2, block3])
    block3 = BatchNormalization()(block3)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2)(block3)
    
    conv = Conv2D(200, (1,1), strides=(1,1))(maxpool)
    avgpool = GlobalAveragePooling2D()(conv)
    
    output = Activation('softmax')(avgpool)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model


def add_layer(inputs, filter_n=128, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=None, use_bias=True):
    '''
    Create a Conv2D, a BatchNormalization, a Activation layer for CNN. 
    Note that, for this project, those three layers together are counted as one layer.
    
    Args:
        @inputs: input of the Conv2D layer
        @filter_n: number of filters in the Conv2D layer
        @activation: activation function for the Activation layer
        @kernel_initializer: kernel_initializer parameter for the Conv2D layer
        @kernel_regularizer: kernel_regularizer parameter for the Conv2D layer 
    Return:
        layer: created layer
    '''
    conv = Conv2D(filter_n, (3,3), strides=(1,1), padding='same', 
                  kernel_initializer=kernel_initializer, 
                  kernel_regularizer=kernel_regularizer, use_bias=use_bias)(inputs)
    bn = BatchNormalization()(conv)
    layer = Activation(activation)(bn)
    return layer

def add_block(inputs, filter_n=[128], activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=None, use_bias=True):
    '''
    Create a block for CNN. A block is made up of several layers.
    Note that, for this project, a Conv2D, a BatchNormalization, a Activation together are counted as one layer.
    
    Args:
        @inputs: input of the first layer in the block
        @filter_n: a list showing the number of filters for each layer
        @activation: activation function for the Activation layer
        @kernel_initializer: kernel_initializer parameter for the Conv2D layer
        @kernel_regularizer: kernel_regularizer parameter for the Conv2D layer 
    Return:
        layer: created layer
    Call: 
        add_layer(inputs, filter_n=128, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=None)
    '''
    # add at least one layer to the block
    block = add_layer(inputs, filter_n[0], activation, kernel_initializer, kernel_regularizer, use_bias)
    # add more layers to the block
    if len(filter_n) > 1:
        for n in filter_n[1:]:
            block = add_layer(block, n, activation, kernel_initializer, kernel_regularizer, use_bias)
    return block

def concate_blocks(block1, block2):
    '''
    Concate the output of two blocks. Use it as the input for the next block or layer
    Args:
        @block1, block2: blocks to be concatenated
    Return:
        concated_block: concatenated block
    '''
    # ensure the dimemsion of two layers are equal
    block1 = space_to_depth(block1, block_size=2)
    # concatenate
    concated_block = Concatenate()([block1, block2])
    return concated_block

def training(model, model_name, train_generator, val_generator, callback, checkpoint_filepath, epochs=5, steps_per_epoch=200):
    '''
    Args:
        @model: model to be trained
        @model_name: model name. Used for saving checkpoint models
        @train_generator: generator of training data
        @val_generator: generator of validation data
        @callback: callback paramter passed to Model.fit()
        @checkpoint_filepath: path for saving checkpoint models
        @epochs: training epochs
        @steps_per_epoch: number of steps in one epoch
    Return:
        model: trained model
        history: history of training result. history.history will return a dictionary of loss, accuracy, val_loss, val_accuracy, lr
    '''
    # If the path does not exist, create it first
    if not os.path.exists(checkpoint_filepath):
        os.mkdir(checkpoint_filepath)
    # make model name = model_name + time
    model._name = '{model}_{time}'.format(model=model_name,
                                          time=datetime.datetime.now().strftime("%m-%d_%H-%M"))
    # create checkpoint ctriteria
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath+'/'+model.name+'.{epoch:02d}'+'.h5', 
                                 monitor='val_accuracy', 
                                 save_best_only=True)
    csv_logger = CSVLogger(checkpoint_filepath+'/'+model.name+'.log', separator=",", append=False)
    # train the model
    history = model.fit(x=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=val_generator, 
                        callbacks=[callback, checkpoint, csv_logger])
    
    return model, history

def save_model(model, model_name, model_path):
    '''
    Save model to a given path with a given name
    Args:
        @model: model to be saved
        @model_name: model name
        @model_path: save path
    '''
    # If the path does not exist, create it first
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # make model name = model_name + time
    model._name = '{model}_{time}'.format(model=model_name,
                                          time=datetime.datetime.now().strftime("%m-%d_%H-%M"))
    # save model
    model.save(model_path+'/'+model.name+'.h5')
    print('{model}.h5 saved at {path}!'.format(model=model.name, path=model_path))

    

def predict(model, generator, model_name, data_path):
    '''
    Predict data from generator and visualize prediction sample
    Args: 
        @model: CNN model
        @generator: generator of testing data
        @model_name: str, name of the model showing on the plot
        @data_path: path where the data is saved
    Return: 
        scores: list, [loss, accuracy]
        true_labels: true labels
        predict_labels: predict labels
    Call: 
        show_prediction(images, labels, labels_pre, model_name)
        get_lookup_tables(path, generator)
        get_labels(path, indexes, generator)
    '''
    
    # evaluate model
    print('Start evaluating...')
    scores = model.evaluate(x=generator)
    print('Loss:{loss:.2f}\nAccuracy:{accuracy:.2f}'.format(loss=scores[0],accuracy=scores[1]))
    
    # predict model using loop to get labels, becuase model.evaluate does not return labels
    print('Start predicting...will take a few minutes')
    # create a list to store labels of each loop
    true_labels = []
    predict_labels = []
    # get lookup table: dictionary {index: class_name}
    lookup_table = get_lookup_tables(data_path, generator)
    
    for i in range(generator.samples // generator.batch_size):
        x, y = generator.next()
        y_pre = model.predict(x=x)
        # generate labels
        y_labels = get_labels(data_path, y, generator)
        y_pre_index = y_pre.argmax(axis=1)
        y_pre_labels = [lookup_table[i] for i in y_pre_index]
        # append result to list
        true_labels += y_labels
        predict_labels += y_pre_labels
    # plot prediction examples
    show_prediction(x, y_labels, y_pre_labels, model_name)
    return scores, true_labels, predict_labels
    