import tensorflow as tf
from src.Net import Net
import numpy as np
import cv2
import time

class Pnet(Net):
    """Pnet网络前向传播"""
    
    def __init__(self, 
                 weight_path, 
                 scales,
                 threshold,
                 nms_threshold):
        """初始化，构建网络架构，读取网络权重"""
        
        super().__init__()
        
        self.__threshold = threshold
        
        self.__nms_threshold = nms_threshold
        
        self.__model = self.__create_model()
        
        self.__model.load_weights(weight_path, by_name=True)
        
        self.__scales = scales
        
        
    #######################################
    # 主函数，前向传播函数，获得并且处理pnet得到的结果
    # 1.获取网络预测值
    # 2.处理预测结果
    #    1).找到大于阈值的位置
    #    2).找到边框的位置
    #    3).获取偏移量
    #    4).获取得分值
    #######################################
    def forward(self, pnet_need_imgs):
        """前向传播函数，主要接口
        
        这个函数将完成pnet的全过程，输入图片
        获得预测值，并对预测值进行处理
        
        Args:
            pnet_need_imgs:(np.array), shape=(x, h, w, 3)
            
        Returns:
            返回处理得到的预测人脸框. (np.array)
        
        """
        self.width = pnet_need_imgs[0].shape[0]
        self.height = pnet_need_imgs[0].shape[1]
        
        rectangles = []
        
        # 传入网络
        out = self.__model.predict(pnet_need_imgs)
        
        # 获取矩形框
        boundingbox = self.__get_boundingbox(out)
   
        if len(boundingbox) == 0:
            
            self.print_messages("该张图像在pnet网络检测不到人脸")
            
            return []
        
        # 将矩形框调整成正方形
        boundingbox = self._rect2square(boundingbox)

        # 避免数值不合理
        boundingbox = self._trimming_frame(boundingbox,
                                           width = self.width,
                                           height = self.height)

        # nms
        boundingbox = self._nms(boundingbox, 0.3)
        
        self.print_messages("Pnet网络处理完毕")

        return self._nms(boundingbox, self.__nms_threshold)
    
    #######################################
    # 找到边框位置
    #######################################
    def __get_boundingbox(self, out):
        """这个方法主要用于判断大于阈值的坐标，并且转换成矩形框
        
        Args:
            cls_prob: pnet网络输出得到的第一个array
        
        """
        
        boundingbox = []
        
        #scores = []
        
        for i in range(len(self.__scales)):
            
            scale = self.__scales[i]
            
            cls_prob = out[0][i, :, :, 1]
            
            (x, y), bbx = self.__boundingbox(cls_prob, scale)

            if bbx.shape[0] == 0:
                continue
                
            scores= np.array(out[0][i, x, y, 1][np.newaxis, :].T)
            
            offset = out[1][i, x, y]*12*(1/scale) 
            
            bbx = bbx + offset
            bbx = np.concatenate((bbx, scores), axis=1)
            
#             # 将矩形框调整成正方形
#             bbx = self._rect2square(bbx)

#             # 避免数值不合理
#             bbx = self._trimming_frame(bbx)

#             # nms
#             bbx = self._nms(bbx, 0.3)            
            
            
            for b in bbx:
                
                boundingbox.append(b)
            
        return np.array(boundingbox)
    
    def __boundingbox(self, cls_prob, scale):
        
        x, y = np.where(cls_prob > self.__threshold)

        bbx = np.array((y, x)).T

        left_top = np.fix(((bbx * 2) + 0) * (1/scale))
        
        right_down = np.fix(((bbx * 2) + 11) * (1/scale))

        return (x, y), np.concatenate((left_top, right_down), axis=1)
  
    #######################################
    # 定义网络架构
    #######################################
    @classmethod
    def __create_model(cls):
        """定义PNet网络的架构"""

        input = tf.keras.Input(shape=[None, None, 3])
        x = tf.keras.layers.Conv2D(10, (3, 3),
                                   strides=1,
                                   padding='valid',
                                   name='conv1')(input)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2],
                                  name='PReLU1')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(16, (3, 3),
                                   strides=1,
                                   padding='valid',
                                   name='conv2')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2],
                                  name='PReLU2')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3),
                                   strides=1, padding='valid', name='conv3')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2],
                                  name='PReLU3')(x)

        classifier = tf.keras.layers.Conv2D(2, (1, 1),
                                            activation='softmax',
                                            name='conv4-1')(x)
        bbox_regress = tf.keras.layers.Conv2D(4, (1, 1),
                                              name='conv4-2')(x)

        model = tf.keras.models.Model([input], [classifier, bbox_regress])

        return model

