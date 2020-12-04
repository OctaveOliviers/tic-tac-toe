# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-12-04 10:31:22
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-12-04 10:45:12

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Dense, Flatten


nrow = 5
ncol = 10

# input is board
input = Input(shape=(nrow,ncol,1))

# convolutional filter
conv  = Conv2D(filters=8, kernel_size=3, strides=1)(input)
bnorm = BatchNormalization()(conv)
relu  = ReLU()(bnorm)

# # residual blocks
# for l in range(3):
#     # input of residual block
#     inres = relu
#     # first convolutional block
#     conv  = Conv2D(filters=8, kernel_size=3, strides=1)(relu)
#     bnorm = BatchNormalization()(conv)
#     relu  = ReLU()(bnorm)
#     # second convolutional block
#     conv  = Conv2D(filters=8, kernel_size=3, strides=1)(relu)
#     bnorm = BatchNormalization()(conv)
#     # residual connection
#     res   = Add()([inres, bnorm])
#     relu  = ReLU()(res)

# build policy head
p_conv  = Conv2D(filters=4, kernel_size=3, strides=1)(relu)
p_bnorm = BatchNormalization()(p_conv)
p_relu  = ReLU()(p_bnorm)
p_head  = Dense(nrow*ncol, activation='softmax')(p_relu)

# build value head
v_conv  = Conv2D(filters=1, kernel_size=3, strides=1)(relu)
v_bnorm = BatchNormalization()(p_conv)
v_relu  = ReLU()(p_bnorm)
v_head  = Dense(1, activation='sigmoid')(p_relu)

# define the model
model = Model(inputs=input, outputs=(p_head, v_head))

model.summary()