import tensorflow as tf
import cv2
import numpy as np
import time

class Net:
    
    def __init__(self, 
                 print_message=True,
                 resize_shape=(80, 80)):
        """初始化"""
        self.print_message = print_message
        self.width = resize_shape[0]
        self.height = resize_shape[1]
    
    
    ########################################
    # 构造打印debug信息的函数以及装饰器，
    # 如果print_message==False，则不打印信息
    ########################################
    def can_show_message(func):
        def inner(self, mess):
            if self.print_message == False:
                pass
            else:
                return func(self, mess)
        return inner
        
    @can_show_message
    def print_messages(self, mess):
        print(mess)
        print("*"*10)
        
        
    #######################################
    # 非极大值抑制
    #######################################
    @staticmethod
    def _nms(rectangles, threshold):
        """非极大值抑制

        经过网络得到的矩形框可能有较多的重叠，我们先取出
        概率最大的框，与其余所有框分别作交并比(IOU), IOU
        在设定阈值以内的框则去除。

        经过nms处理，可以去除重合度相对高的矩形框，保留得
        分概率比较高，并且重合度相对低的矩形框

        Args:
            rectangles: 所有矩形框的坐标值以及得分 (numpy.ndarray)
                For example：
                 [x1,     y1,    x2,    y2,     score]
                [[132.    17.   138.    23.     0.83198518]
                 [133.    17.   139.    23.     0.5036146 ]
                 [138.    20.   144.    26.     0.533454  ]
                 [117.    28.   123.    34.     0.50165093]]

            threshold: 阈值 (float)

        Returns:
            经过nms处理后剩余的矩形框的坐标值以及概率得分(numpy.ndarray)
                For example：
                 [x1,     y1,    x2,    y2,     score]
                [[132.    17.   138.    23.     0.83198518]
                 [117.    28.   123.    34.     0.50165093]]

        """
        if len(rectangles)==0:
            return rectangles
        boxes = np.array(rectangles)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s  = boxes[:, 4]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = np.array(s.argsort())
        pick = []
        while len(I)>0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where(o<=threshold)[0]]
        result_rectangle = boxes[pick].tolist()
        return result_rectangle

    
    #######################################
    # 将矩形框修整为正方形
    #######################################
    @staticmethod
    def _rect2square(rectangles):
        """将矩形框修整为正方形

        经过pnet或者rnet得到的矩形框大多为长方形，但是rnet或者onet
        网络的输入格式是正方形，因此我们取长方形较长的边，作为矫正后
        正方形的边，得到的正方形区域会比原来的矩形框稍大。

        Args:
            rectangles: 矩形框的坐标值
                For example：
                 [x1,     y1,    x2,    y2,     score]
                [[132.    17.   138.    23.     0.83198518]
                 [117.    28.   123.    34.     0.50165093]]

        Returns:
            与输入的类型以及大小都相同

        """
        rectangles = np.array(rectangles)
        w = rectangles[:,2] - rectangles[:,0]
        h = rectangles[:,3] - rectangles[:,1]
        l = np.maximum(w,h).T
        rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
        rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5
        rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T
        return rectangles

    
    #######################################
    # 对矩形框进行调整，避免出现数值不合理的情况
    #######################################
    def _trimming_frame(self, rectangles, width, height):
        """对矩形框进行调整，避免出现数值不合理的情况

        无论经过网络得到的矩形框还是其他处理后得到的矩形框，
        值有可能不合理(例如小于零或者大于原图像的长宽)，此
        方法的目的就是使矩形框不超过合理范围

        Args:
            rectangles: 矩形框的坐标值
                For example：
                 [x1,     y1,    x2,    y2,     score]
                [[132.    17.   138.    23.     0.83198518]
                 [117.    28.   123.    34.     0.50165093]]

        Returns:
            与输入的类型以及大小都相同

        """
        for j in range(len(rectangles)):

            rectangles[j][0] = max(0, int(rectangles[j][0]))
            rectangles[j][1] = max(0, int(rectangles[j][1]))
            rectangles[j][2] = min(width, int(rectangles[j][2]))
            rectangles[j][3] = min(height, int(rectangles[j][3]))
            
            if rectangles[j][0] >= rectangles[j][2]:
                rectangles[j][0] = 0
            elif rectangles[j][1] > rectangles[j][3]:
                rectangles[j][1] = 0

        return rectangles
