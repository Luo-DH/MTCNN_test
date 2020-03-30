Help on module Onet:

NAME
    Onet

CLASSES
    Net.Net(builtins.object)
        Onet
    
    class Onet(Net.Net)
     |  Method resolution order:
     |      Onet
     |      Net.Net
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, weight_path, threshold, nms_threshold)
     |      初始化，构建网络架构，读取网络权重
     |  
     |  forward(self, onet_need_imgs, rnet_got_rects, image)
     |      核心函数，前向网络传播
     |      
     |      将输入数据传入网络中，得到预测人脸框的坐标值以及
     |      五官的五个坐标值，以数组形式返回，方便用户使用
     |      
     |      Returns:
     |          左上角与右下角的坐标值(x, y). For example:
     |      
     |          array([[134.,  19., 138.,  23.],
     |                 [ 34.,  86.,  64., 116.],
     |                 [ 11.,  18.,  99., 107.]])
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Net.Net:
     |  
     |  can_show_message(func)
     |      ########################################
     |      # 构造打印debug信息的函数以及装饰器，
     |      # 如果print_message==False，则不打印信息
     |      ########################################
     |  
     |  print_messages = inner(self, mess)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from Net.Net:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FILE
    /home/luo/myself/2020-3-30/src/Onet.py


