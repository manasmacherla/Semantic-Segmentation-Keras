import cv2
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def vgg_encoder(input_height = 224, input_width = 224, weights = 'imagenet'):
    img_input = Input(input_height, input_width, 3)
    x = Conv2D(64, (3,3), activation = 'relu', padding = 'same', name = 'conv1_layer1')(img_input)
    x = Conv2D(64, (3,3), activation = 'relu', padding = 'same', name = 'conv2_layer1')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool_layer1')(x)

    layer1 = x

    x = Conv2D(128, (3,3), activation = 'relu', padding = 'same', name = 'conv1_layer2')(x)
    x = Conv2D(128, (3,3), activation = 'relu', padding = 'same', name = 'conv2_layer2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool_layer2')(x)

    layer2 = x

    x = Conv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'conv1_layer3')(x)
    x = Conv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'conv2_layer3')(x)
    x = Conv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'conv3_layer3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool_layer3')(x)

    layer3 = x

    x = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'conv1_layer4')(x)
    x = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'conv2_layer4')(x)
    x = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'conv3_layer4')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool_layer4')(x)

    layer4 = x

    x = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'conv1_layer5')(x)
    x = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'conv2_layer5')(x)
    x = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'conv3_layer5')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool_layer5')(x)

    layer5 = x

    



