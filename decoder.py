from model import vgg_encoder
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def decoder(classes, input_height = 416, input_width = 608):

    img_input, layers = vgg_encoder(input_height=416, input_width=608)
    [l1, l2, l3, l4, l5] = layers

    y = l5

    y = Conv2D(4096, (7,7), activation= 'relu', padding='same')(y)
    y = Dropout(0.5)(y)
    y = Conv2D(4096, (1,1), activation= 'relu', padding='same')(y)
    y = Dropout(0.5)(y)
    y = Conv2D(classes, (1,1), kernel_initializer='he_normal')(y)    

    y = Conv2DTranspose(classes, kernel_size=(4,4), stride=(2,2), use_bias=False)(y)
    layer4 = l4
    layer4 = Conv2D(classes, (1,1), kernel_initializer='he_normal')
    y = Add()([y, layer4]) #they are FCN-16

    y = Conv2DTranspose(classes, kernel_size=(4,4), stride=(2,2), use_bias=False)(y)
    layer3 = l3
    layer3 = Conv2D(classes, (1,1), kernel_initializer='he_normal')
    y = Add()([y, layer3])

    y = Conv2DTranspose(classes, kernel_size=(16,16), stride=(8,8), use_bias=False)(y)

    y = (Activation('softmax'))(y)

    return y





