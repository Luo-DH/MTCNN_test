import tensorflow as tf
from src.Net import Net
import numpy as np
import cv2
import time


class Onet(Net):
    
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


    def __get_landmark(self, outs, rnet_got_rects):

        classifier = outs[0]
        x = np.where(classifier[:, 1] > self.__threshold)

        onet_pts = outs[2]

        offset_x1 = onet_pts[x, 0]
        offset_y1 = onet_pts[x, 5]
        offset_x2 = onet_pts[x, 1]
        offset_y2 = onet_pts[x, 6]
        offset_x3 = onet_pts[x, 2]
        offset_y3 = onet_pts[x, 7]
        offset_x4 = onet_pts[x, 3]
        offset_y4 = onet_pts[x, 8]
        offset_x5 = onet_pts[x, 4]
        offset_y5 = onet_pts[x, 9]

        x1 = rnet_got_rects[0][0]
        y1 = rnet_got_rects[0][1]
        x2 = rnet_got_rects[0][2]
        y2 = rnet_got_rects[0][3]

        w = x2 - x1
        h = y2 - y1

        onet_pts_x1 = np.array(offset_x1*w + x1)
        onet_pts_x2 = np.array(offset_x2*w + x1)
        onet_pts_x3 = np.array(offset_x3*w + x1)
        onet_pts_x4 = np.array(offset_x4*w + x1)
        onet_pts_x5 = np.array(offset_x5*w + x1)
        onet_pts_y1 = np.array(offset_y1*h + y1)
        onet_pts_y2 = np.array(offset_y2*h + y1)
        onet_pts_y3 = np.array(offset_y3*h + y1)
        onet_pts_y4 = np.array(offset_y4*h + y1)
        onet_pts_y5 = np.array(offset_y5*h + y1)

        onet_left_eye = np.concatenate((onet_pts_x1, 
                                        onet_pts_y1), axis=1)
        onet_right_eye = np.concatenate((onet_pts_x2, 
                                         onet_pts_y2), axis=1)
        onet_nose = np.concatenate((onet_pts_x3, 
                                    onet_pts_y3), axis=1)
        onet_left_mouth = np.concatenate((onet_pts_x4, 
                                          onet_pts_y4), axis=1)
        onet_right_mouth = np.concatenate((onet_pts_x5, 
                                           onet_pts_y5), axis=1)

        return (onet_left_eye, 
                onet_right_eye, 
                onet_nose, 
                onet_left_mouth, 
                onet_right_mouth)

    def __get_boundingbox(self, outs, rnet_got_rects):
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

        rnet_got_rects = np.array(rnet_got_rects)

        x1 = np.array(rnet_got_rects[x][:, 0])[np.newaxis, :].T
        y1 = np.array(rnet_got_rects[x][:, 1])[np.newaxis, :].T
        x2 = np.array(rnet_got_rects[x][:, 2])[np.newaxis, :].T
        y2 = np.array(rnet_got_rects[x][:, 3])[np.newaxis, :].T

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
    
    
    def forward(self, onet_need_imgs, rnet_got_rects, image):
        """核心函数，前向网络传播

        将输入数据传入网络中，得到预测人脸框的坐标值以及
        五官的五个坐标值，以数组形式返回，方便用户使用

        Returns:
            左上角与右下角的坐标值(x, y). For example:

            array([[134.,  19., 138.,  23.],
                   [ 34.,  86.,  64., 116.],
                   [ 11.,  18.,  99., 107.]])

        """

        self.print_messages("开始通过Rnet网络进行处理")

        outs = self.__model.predict(onet_need_imgs)

        boundingbox = self.__get_boundingbox(outs, rnet_got_rects)

        boundingbox = self._rect2square(boundingbox)

        boundingbox = self._trimming_frame(boundingbox,
                                           width = image.shape[0],
                                           height = image.shape[1])

        landmark = self.__get_landmark(outs, rnet_got_rects)

        return boundingbox, landmark
        
    @classmethod
    def __create_model(cls):
        """定义ONet网络的架构"""

        input = tf.keras.layers.Input(shape = [48,48,3])
        # 48,48,3 -> 23,23,32
        x = tf.keras.layers.Conv2D(32, (3, 3),
                                   strides=1, 
                                   padding='valid', 
                                   name='conv1')(input)
        x = tf.keras.layers.PReLU(shared_axes=[1,2],
                                  name='prelu1')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, 
                                      strides=2, 
                                      padding='same')(x)
        # 23,23,32 -> 10,10,64
        x = tf.keras.layers.Conv2D(64, (3, 3), 
                                   strides=1, 
                                   padding='valid', 
                                   name='conv2')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2],
                                  name='prelu2')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, 
                                      strides=2)(x)
        # 8,8,64 -> 4,4,64
        x = tf.keras.layers.Conv2D(64, (3, 3), 
                                   strides=1, 
                                   padding='valid', 
                                   name='conv3')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2],
                                  name='prelu3')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
        # 4,4,64 -> 3,3,128
        x = tf.keras.layers.Conv2D(128, (2, 2), 
                                   strides=1, 
                                   padding='valid', 
                                   name='conv4')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2],
                                  name='prelu4')(x)
        # 3,3,128 -> 128,12,12
        x = tf.keras.layers.Permute((3,2,1))(x)

        # 1152 -> 256
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, name='conv5') (x)
        x = tf.keras.layers.PReLU(name='prelu5')(x)

        # 鉴别
        # 256 -> 2 256 -> 4 256 -> 10
        classifier = tf.keras.layers.Dense(2, 
                                           activation='softmax',
                                           name='conv6-1')(x)
        bbox_regress = tf.keras.layers.Dense(4,name='conv6-2')(x)
        landmark_regress = tf.keras.layers.Dense(10,name='conv6-3')(x)

        model = tf.keras.models.Model([input], [classifier, bbox_regress, landmark_regress])

        return model
