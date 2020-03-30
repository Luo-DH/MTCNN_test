Help on module MTCNN:

NAME
    MTCNN

CLASSES
    builtins.object
        MTCNN
    
    class MTCNN(builtins.object)
     |  MTCNN，人脸检测，关键点定位
     |  
     |  作用：
     |      1.对输入的图片进行人脸检测，如果图片中
     |      存在人脸，则输出人脸框的预测位置
     |      2.输出人脸关键点的坐标，(left eyes),
     |      (right eyes), (nose), (left mouth),
     |      (right mouth)
     |  
     |  Args:
     |      threshold (list, float, optional): 包含三个浮点型数
     |          的列表，代表pnet,rnet,onet网络的置信度. 如果人脸检测效果
     |          不太好，可以适度调整该参数，检测多过多人脸，或者误将
     |          非人脸当作人脸，可以调大list中的三个数，同理，如果检
     |          测成功率较低，可以调低三个数，调整时，优先调整第一个
     |          数，其次是第二个，最后是第三个数.
     |      nms_threshold (list, float, optional): 包含三个浮点型数
     |          的列表，用以对三个网络的非极大值抑制设定阈值.一般不需要对
     |          该参数作出过多调整如果检测成功率较低，调整threshold参数，
     |          如果能够检测出人脸，不过人脸框的大小位置有问题可以调整该参数，
     |          一般情况下，数字调大，人脸框减少，数字调小，人脸框变多.
     |          Note: 调整此参数，可能会对检测速度产生一定影响.
     |      weight_paths (list, string, optional): pnet,rnet,onet
     |          网络预训练权重的文件位置，需要".h5"格式的文件。如果不提供
     |          该参数，则去默认路径("./weight_paths")寻找，找不到则报错
     |      max_face (bool, optional): 是否只检测最大人脸，True则只会找
     |          到图片中的最大人脸，False会尽可能找到图片中所有的人脸。
     |      save_face (bool, optional): 是否保存检测到的人脸区域，True
     |          则会保存检测提取出人脸区域，并且保存在本地，保存目录需要设置，
     |          False则不保存人脸。
     |      save_dirt (string optional): 如果设置了save_face=True,
     |          则一定需要设置该值，该变量代表保存人脸图片的位置。
     |      print_time (bool optional): 是否格式化输出检测到人脸的时间。
     |          True for yes. False for no.
     |      print_message (bool optional): 是否打印辅助信息。辅助信息
     |          包括程序现在进行的步骤，以及一些额外信息，实际运用时候，可
     |          以设置为false. 设置为false后只会打印基本信息。
     |      detect_hat (bool optional): 是否检测安全帽的佩戴，根据实际
     |          情况选择是否开启。开启后会检测人脸是否有佩戴安全帽，会打印
     |          相关信息。
     |      resize_type ('with_loss/without_loss', optional): 选择
     |          resize图像的方式。with_loss：直接缩放图片大小，会造成长
     |          方形图片扭曲失真。without_loss：以边缘填充的方式resize
     |          图片，图片会边大，不过保留了图片完整信息。
     |      resize_shape (tuple int): 用户输入的图片尺寸。
     |      padding_type ('mask/nomask', optional): 选择pnet网络获取
     |          图片的填充格式。
     |      factor (float, optional): pnet网络需要图片的缩放比例，默认为
     |          0.709,无特殊需求，一般不修改。修改可能会造成速度与准确度的降低
     |      
     |      
     |  Returns:
     |      [], [], bool
     |      rectangles (list, float): 返回检测到的人脸框的左上角坐标和右下角
     |          坐标。如果未检测到人脸框，返回空的list
     |      landmark (list, float): 返回检测到的左眼，右眼，鼻子，左嘴，右嘴
     |          的坐标。如果为检测到人脸，返回空的list
     |      have_face (bool): 如果检测到人脸，返回True, 否则返回False
     |      init_shape (tuple int optional): 图片的初始化大小
     |  
     |  Note: Requires Tensorflow v2.0 or later.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, threshold=[0.5, 0.6, 0.7], nms_threshold=[0.7, 0.7, 0.7], weight_paths=['./weight_path/pnet.h5', './weight_path/rnet.h5', './weight_path/onet.h5'], max_face=False, save_face=False, save_dirt='./', print_time=True, print_message=True, detect_hat=False, resize_type='without_loss', resize_shape=(80, 80), padding_type='mask', factor=0.709)
     |      MTCNN初始化函数
     |  
     |  can_show_message(func)
     |      ########################################
     |      # 构造打印debug信息的函数以及装饰器，
     |      # 如果print_message==False，则不打印信息
     |      #######################################
     |  
     |  detect_face(self, image)
     |      该函数用于人脸检测，对用户主要接口
     |      
     |      用户只需要调用该方法，并且传入需要检测人脸的
     |      图片，需要类型为RGB的三通道图片(opencv读取
     |      的图片需要作通道转换),并且图片的最小尺寸要大
     |      于12x12
     |      
     |      Args:
     |          image(dtype=np.ndarray): 用户传入的
     |              待检测人脸的图片。   
     |                  shape = (x, x, 3) RGB顺序
     |      
     |      Returns:
     |          [], [] , bool 返回矩形框坐标和五官坐标
     |          如果没有检测到人脸，则返回两个空list和False
     |  
     |  fix_rects(self, rects)
     |      ######################################
     |      # 将得到的边界还原到原图合适的比例
     |      ######################################
     |  
     |  get_message(self, image)
     |      ######################################
     |      # 保存图片信息
     |      ######################################
     |  
     |  get_onet_need_imgs(self, image, rects)
     |      获得一系列onet需要的格式的图片
     |      
     |      Args:
     |          rects: 经过rnet处理得到的所有矩形框
     |  
     |  get_rnet_need_imgs(self, image, rects)
     |      获得一系列rnet需要的格式的图片
     |      
     |      Args:
     |          rects: 经过pnet处理得到的所有矩形框
     |  
     |  image_resize(self, image, size)
     |      图像缩放
     |  
     |  image_resize_padding(self, image, size)
     |      缩放函数
     |      
     |      Args:
     |          image: 待缩放的原始图像
     |          size: (tuple, (int, int)),缩放后的尺寸
     |  
     |  mask_template(self, shape)
     |      图片掩码模板
     |      
     |      根据用户输入resize图片的尺寸，
     |      制作模板，方便获取不同大小的pnet
     |      图片的需求
     |  
     |  need_to_detect_hat(func)
     |  
     |  need_to_save_face(func)
     |      ######################################
     |      # 根据需要截取图片，保存到本地
     |      ######################################
     |  
     |  onet_processing(self, rects, image)
     |      封装rnet需要的函数
     |  
     |  pnet_processing(self, image)
     |      封装pnet需要的函数
     |  
     |  print_messages = inner(self, mess)
     |  
     |  rnet_processiong(self, rects, image)
     |      封装rnet需要的函数
     |  
     |  to_get_all_faces(self, rect, image)
     |      获得图像中的所有人脸
     |  
     |  to_get_hat = inner(self, rects, image)
     |  
     |  to_get_max_face(self, rects, image)
     |      获取图像中的最大人脸
     |  
     |  to_save_face = inner(self, rects, image, dirt)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FILE
    /home/luo/myself/2020-3-30/src/MTCNN.py


