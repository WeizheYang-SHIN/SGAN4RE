# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2022/3/9 17:20 
# @Author : wz.yang 
# @File : resNet.py
# @desc :

def res_block_v1(x, input_filter, output_filter):
    res_x = Conv2D(kernel_size=(3, 3), filters=output_filter, strides=1, padding='same')(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation('relu')(res_x)
    res_x = Conv2D(kernel_size=(3, 3), filters=output_filter, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)
    if input_filter == output_filter:
        identity = x
    else:  # 需要升维或者降维
        identity = Conv2D(kernel_size=(1, 1), filters=output_filter, strides=1, padding='same')(x)
    x = keras.layers.add([identity, res_x])
    output = Activation('relu')(x)
    return output
