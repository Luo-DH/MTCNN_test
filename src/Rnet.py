import tensorflow as tf

from src.Net import Net
import cv2
import numpy as np
import time


class Rnet(Net):
        
    def __init__(self, 
                 weight_path, 
                 threshold,
                 nms_threshold):
        """初始化，构建网络架构，读取网络权重"""
        
        super().__init__()
        
        self.__threshold = threshold
        
        self.__nms_threshold = nms_threshold
        
        self.__model = self.__create_model()
        
        self.__model.load_weights(weight_path, by_name=True)
       

    def __get_boundingbox(self, outs, pnet_got_rects):
        """这个函数用于得到加上偏移后的矩形框坐标

        Args:
            outs: 经过网络后得到的结果

        Attributes:

            offset: 经过网络后得到的偏移量. For example:

                [[[[ 0.0463337   0.01343044 -0.125744   -0.03012199]]

                  [[-0.04171417 -0.19884819  0.18158348  0.25635445]]]]

            x: 符合阈值条件的坐标值 For example:

                array([ 3, 11]),

        Returns: 加上偏移后的坐标值

            [[112.92667393  20.26860878 129.48512     39.39756013]

             [  6.28743928   0.30251127 115.16092964 129.81554615]]

        """

        # 人脸概率
        classifier = outs[0]

        # 偏移量
        offset = outs[1]

        x = np.where(classifier[:, 1] > self.__threshold)

        # 获得相应位置的offset值
        offset = offset[x, None]

        dx1 = np.array(offset[0])[:, :, 0]
        dy1 = np.array(offset[0])[:, :, 1]
        dx2 = np.array(offset[0])[:, :, 2]
        dy2 = np.array(offset[0])[:, :, 3]

        pnet_got_rects = np.array(pnet_got_rects)

        x1 = np.array(pnet_got_rects[x][:, 0])[np.newaxis, :].T
        y1 = np.array(pnet_got_rects[x][:, 1])[np.newaxis, :].T
        x2 = np.array(pnet_got_rects[x][:, 2])[np.newaxis, :].T
        y2 = np.array(pnet_got_rects[x][:, 3])[np.newaxis, :].T

        w = x2 - x1
        h = y2 - y1

        new_x1 = np.fix(x1 + dx1*w)
        new_x2 = np.fix(x2 + dx2*w)
        new_y1 = np.fix(y1 + dy1*h)
        new_y2 = np.fix(y2 + dy2*h)

        score = np.array(classifier[x, 1]).T


        boundingbox = np.concatenate((new_x1, 
                                      new_y1, 
                                      new_x2, 
                                      new_y2, 
                                      score), axis=1)

        return boundingbox
        

    def forward(self, rnet_need_imgs, pnet_got_rects, image):
        """核心函数，前向网络传播

        将输入数据传入网络中，并处理得到的结果

        Args:
            rects: 将pnet网络得到的预测人脸框提取出来，
                resize成24x24，作为rnet的输入

        Returns:
            左上角与右下角的坐标值(x, y). For example:

                [[125.0, 272.0, 142.0, 289.0, 0.9670179486274719],
                [123.0, 269.0, 145.0, 291.0, 0.9095805287361145],
                [79.0, 305.0, 91.0, 317.0, 0.8930739760398865]]

        """

        self.print_messages("开始通过Rnet网络进行处理")

        rnet_need_imgs = np.array(rnet_need_imgs)

        outs = self.__model.predict(rnet_need_imgs)

        boundingbox = self.__get_boundingbox(outs, pnet_got_rects)

        # 将矩形框变成正方形
        boundingbox = self._rect2square(boundingbox)

        # 微调，避免数值不合理
        boundingbox = self._trimming_frame(boundingbox, 
                                           image.shape[0],
                                           image.shape[1])

        # nms
        boundingbox = self._nms(boundingbox, self.__nms_threshold)
        
        return boundingbox

    @classmethod
    def __create_model(cls):
        """定义RNet网络的架构"""

        input = tf.keras.Input(shape=[24, 24, 3])
        x = tf.keras.layers.Conv2D(28, (3, 3), 
                                   strides=1, 
                                   padding='valid', 
                                   name='conv1')(input)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], 
                                  name='prelu1')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=3, 
                                         strides=2, 
                                         padding='same')(x)

        x = tf.keras.layers.Conv2D(48, (3, 3), 
                                   strides=1, 
                                   padding='valid', 
                                   name='conv2')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], 
                                  name='prelu2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=3, 
                                         strides=2)(x)

        x = tf.keras.layers.Conv2D(64, (2, 2), 
                                   strides=1, 
                                   padding='valid', 
                                   name='conv3')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], 
                                  name='prelu3')(x)

        x = tf.keras.layers.Permute((3, 2, 1))(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(128, name='conv4')(x)
        x = tf.keras.layers.PReLU(name='prelu4')(x)

        classifier = tf.keras.layers.Dense(2, 
                                           activation='softmax', 
                                           name='conv5-1')(x)
        bbox_regress = tf.keras.layers.Dense(4, name='conv5-2')(x)

        model = tf.keras.models.Model([input], [classifier, bbox_regress])

        return model
