Help on module Rnet:

NAME
    Rnet

CLASSES
    Net.Net(builtins.object)
        Rnet
    
    class Rnet(Net.Net)
     |  Method resolution order:
     |      Rnet
     |      Net.Net
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, weight_path, threshold, nms_threshold)
     |      初始化，构建网络架构，读取网络权重
     |  
     |  forward(self, rnet_need_imgs, pnet_got_rects, image)
     |      核心函数，前向网络传播
     |      
     |      将输入数据传入网络中，并处理得到的结果
     |      
     |      Args:
     |          rects: 将pnet网络得到的预测人脸框提取出来，
     |              resize成24x24，作为rnet的输入
     |      
     |      Returns:
     |          左上角与右下角的坐标值(x, y). For example:
     |      
     |              [[125.0, 272.0, 142.0, 289.0, 0.9670179486274719],
     |              [123.0, 269.0, 145.0, 291.0, 0.9095805287361145],
     |              [79.0, 305.0, 91.0, 317.0, 0.8930739760398865]]
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
    /home/luo/myself/2020-3-30/src/Rnet.py


