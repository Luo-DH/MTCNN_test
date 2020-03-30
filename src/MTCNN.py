import tensorflow as tf
from src.Net import Net
from src.Pnet import Pnet
from src.Rnet import Rnet
from src.Onet import Onet
import numpy as np
import time
import cv2

class MTCNN:
    """MTCNN，人脸检测，关键点定位
    
    作用：
        1.对输入的图片进行人脸检测，如果图片中
        存在人脸，则输出人脸框的预测位置
        2.输出人脸关键点的坐标，(left eyes),
        (right eyes), (nose), (left mouth),
        (right mouth)
    
    Args:
        threshold (list, float, optional): 包含三个浮点型数
            的列表，代表pnet,rnet,onet网络的置信度. 如果人脸检测效果
            不太好，可以适度调整该参数，检测多过多人脸，或者误将
            非人脸当作人脸，可以调大list中的三个数，同理，如果检
            测成功率较低，可以调低三个数，调整时，优先调整第一个
            数，其次是第二个，最后是第三个数.
        nms_threshold (list, float, optional): 包含三个浮点型数
            的列表，用以对三个网络的非极大值抑制设定阈值.一般不需要对
            该参数作出过多调整如果检测成功率较低，调整threshold参数，
            如果能够检测出人脸，不过人脸框的大小位置有问题可以调整该参数，
            一般情况下，数字调大，人脸框减少，数字调小，人脸框变多.
            Note: 调整此参数，可能会对检测速度产生一定影响.
        weight_paths (list, string, optional): pnet,rnet,onet
            网络预训练权重的文件位置，需要".h5"格式的文件。如果不提供
            该参数，则去默认路径("./weight_paths")寻找，找不到则报错
        max_face (bool, optional): 是否只检测最大人脸，True则只会找
            到图片中的最大人脸，False会尽可能找到图片中所有的人脸。
        save_face (bool, optional): 是否保存检测到的人脸区域，True
            则会保存检测提取出人脸区域，并且保存在本地，保存目录需要设置，
            False则不保存人脸。
        save_dirt (string optional): 如果设置了save_face=True,
            则一定需要设置该值，该变量代表保存人脸图片的位置。
        print_time (bool optional): 是否格式化输出检测到人脸的时间。
            True for yes. False for no.
        print_message (bool optional): 是否打印辅助信息。辅助信息
            包括程序现在进行的步骤，以及一些额外信息，实际运用时候，可
            以设置为false. 设置为false后只会打印基本信息。
        detect_hat (bool optional): 是否检测安全帽的佩戴，根据实际
            情况选择是否开启。开启后会检测人脸是否有佩戴安全帽，会打印
            相关信息。
        resize_type ('with_loss/without_loss', optional): 选择
            resize图像的方式。with_loss：直接缩放图片大小，会造成长
            方形图片扭曲失真。without_loss：以边缘填充的方式resize
            图片，图片会边大，不过保留了图片完整信息。
        resize_shape (tuple int): 用户输入的图片尺寸。
        padding_type ('mask/nomask', optional): 选择pnet网络获取
            图片的填充格式。
        factor (float, optional): pnet网络需要图片的缩放比例，默认为
            0.709,无特殊需求，一般不修改。修改可能会造成速度与准确度的降低
        
        
    Returns:
        [], [], bool
        rectangles (list, float): 返回检测到的人脸框的左上角坐标和右下角
            坐标。如果未检测到人脸框，返回空的list
        landmark (list, float): 返回检测到的左眼，右眼，鼻子，左嘴，右嘴
            的坐标。如果为检测到人脸，返回空的list
        have_face (bool): 如果检测到人脸，返回True, 否则返回False
        init_shape (tuple int optional): 图片的初始化大小

    Note: Requires Tensorflow v2.0 or later.  
    
    """
    
    def __init__(self,
                 threshold=[0.5, 0.6, 0.7],
                 nms_threshold=[0.7, 0.7, 0.7],
                 weight_paths=['./weight_path/pnet.h5',
                               './weight_path/rnet.h5',
                               './weight_path/onet.h5'],
                 max_face=False,
                 save_face=False,
                 save_dirt="./",
                 print_time=True,
                 print_message=True,
                 detect_hat=False,
                 resize_type='without_loss',
                 resize_shape=(80, 80),
                 padding_type='mask',
                 factor=0.709):
        """MTCNN初始化函数"""
        
        print("MTCNN TEXT VERSION:2.1 BY Luo-DH, NBUT")
        
        self.threshold = threshold # 三个网络的阈值
        self.nms_threshold = nms_threshold # nms的阈值
        self.weight_paths = weight_paths # 权重保存的路径
        self.max_face = max_face # 是否只检测最大人脸
        self.save_face = save_face # 是否保存检查到的人脸
        self.save_dirt = save_dirt # 设置保存图像的路径
        self.print_time = print_time # 是否打印检测到人脸的时间
        self.print_message = print_message # 是否打印debug信息
        self.detect_hat = detect_hat # 是否检测安全帽
        self.resize_type = resize_type # resize图像的方式
        self.resize_shape = resize_shape # 初始化缩放的尺寸
        self.padding_type = padding_type # 填充方式
        self.__factor = factor # pnet图像的缩放比例
        
        # 获得一系列的缩放系数
        self.__scales = self.__get_scales()
        
        self.__net = Net(self.print_message,
                         self.resize_shape)
        
        self.__pnet = Pnet(self.weight_paths[0],
                           self.__scales,
                           self.threshold[0],
                           self.nms_threshold[0])
        
        self.__rnet = Rnet(self.weight_paths[1],
                           self.threshold[1],
                           self.nms_threshold[1])

        self.__onet = Onet(self.weight_paths[2],
                           self.threshold[2],
                           self.nms_threshold[2])
        
    ########################################
    # 获得一系列的比例系数
    ########################################
    def __get_scales(self):
        """这个函数用于获得缩放比例
        
        将原始图片进行缩放，保存缩放系数，保证缩小成最小值
        后，长和宽仍然大于12，否则无法传入pnet网络
        
        Args:
            shape: 输入图片的长和宽 (int, int)
            
        Returns:
            缩放比例，list. 
            Fof example: []
        
        """
        
        i = 0
        
        scales = []

        while True:

            scale = self.__factor ** i

            tmp_width = self.resize_shape[0] * scale
            tmp_height = self.resize_shape[1] * scale

            # 如果缩放成小于12，则不符合要求
            if min(tmp_width, tmp_height) <= 13:

                break

            scales.append(scale) # 符合要求的值放入__scale中
            i += 1 # i的值每次加一，以便减小scale的值
            
        return scales
        
    ########################################
    # 构造打印debug信息的函数以及装饰器，
    # 如果print_message==False，则不打印信息
    #######################################
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

    ######################################
    # 保存图片信息
    ######################################
    def get_message(self, image):
        width = image.shape[0]
        height = image.shape[1]

        big_side = width if width>height else height
        
        self.width_scale = big_side / self.resize_shape[0]
        self.height_scale = big_side / self.resize_shape[1]
        
        
        
    ######################################
    # 主函数，对用户检测人脸提供接口
    ######################################
    def detect_face(self, image):
        """该函数用于人脸检测，对用户主要接口
        
        用户只需要调用该方法，并且传入需要检测人脸的
        图片，需要类型为RGB的三通道图片(opencv读取
        的图片需要作通道转换),并且图片的最小尺寸要大
        于12x12
        
        Args:
            image(dtype=np.ndarray): 用户传入的
                待检测人脸的图片。   
                    shape = (x, x, 3) RGB顺序
        
        Returns:
            [], [] , bool 返回矩形框坐标和五官坐标
            如果没有检测到人脸，则返回两个空list和False
        
        """
        
        # 预处理，为最后还原信息使用
        self.get_message(image)
        
        # pnet网络处理
        rects, ret = self.pnet_processing(image.copy())
        
        if ret == False:
            return [], [], False
        
        # rnet网络处理
        rects, ret = self.rnet_processiong(rects, image.copy())
  
        if ret == False:
            return [], [], False
        
        # onet网络处理
        rects, landmark, ret = self.onet_processing(rects, 
                                               image.copy())
        
        if ret == False:
            return [], [], False
        
        # 边界框最后调整
        self.fix_rects(rects)
        
        # 如果保存图片
        self.to_save_face(rects, image.copy(), self.save_dirt) 
        
        # 如果需要获取安全帽
        self.to_get_hat(rects, image.copy())

        return rects, landmark, ret
        
    ######################################
    # 将得到的边界还原到原图合适的比例
    ######################################
    def fix_rects(self, rects):
        
        for rect in rects:

            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            rect[0] = rect[0] * self.width_scale
            rect[1] = rect[1] * self.height_scale
            rect[2] = rect[0] + width * self.width_scale
            rect[3] = rect[1] + height * self.height_scale
    
            
    ######################################
    # 根据需要截取图片，保存到本地
    ######################################
    def need_to_save_face(func):
        def inner(self, rects, image, dirt):
            
            if self.save_face == False:
                pass
            else:
                return func(self, rects, image, dirt)
        return inner
    
    @need_to_save_face
    def to_save_face(self, rects, image, dirt):
        
        name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + str(time.clock())
        
        if self.max_face == True:
            
            img_ = self.to_get_max_face(rects, image)
            
            if(cv2.imwrite(dirt+"/"+name+".jpg", img_)):
                    self.print_messages("成功保存人脸到{}".format(dirt+"/"+name+".jpg"))
            
            
        else:
            
            for i in range(len(rects)):
            
                # 获取每个矩形框
                rect = rects[i]

                img_ = self.to_get_all_faces(rect, image)
                #show_img(img_)
                if(cv2.imwrite(dirt+"/"+name+".jpg", img_)):
                    self.print_messages("成功保存人脸到{}".format(dirt+"/"+name+".jpg"))
    
    
    def to_get_all_faces(self, rect, image):
        """获得图像中的所有人脸"""
        img_ = image.copy()[int(rect[1]): int(rect[3]),
                            int(rect[0]): int(rect[2])]
        
        return img_
    
    def to_get_max_face(self, rects, image):
        """获取图像中的最大人脸"""
        areas = []
        for rect in rects:
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            area = width*height
            
            areas.append(area)
            
        index = np.argmax(np.array(areas), axis=0) #竖着比较，返回行号
        
        img_ = self.to_get_all_faces(rects[index], image)
        
        return img_
    
    ######################################
    # 获取安全帽
    ######################################
    
    def need_to_detect_hat(func):
        def inner(self, rects, image):
            
            if self.detect_hat == False:
                pass
            else:
                return func(self, rects, image)
        return inner
    
    @need_to_detect_hat
    def to_get_hat(self, rects, image):
        
        for rect in rects:
            
            height = rect[3] - rect[1]
            
            rect[1] = rect[3] - 1.5*height
            
        rects = self.__net._rect2square(rects)
        
        rects = self.__net._trimming_frame(rects,
                                   width = image.shape[0],
                                   height = image.shape[1])   
        print(rects[0][3]-rects[0][1])
        print(rects[0][2]-rects[0][0])
        # 保存带有安全帽的人脸路径
        dirt = self.save_dirt + "/hat"
        
        self.to_save_face(rects, image, dirt)
        

        
        
    ######################################
    # 定义pnet网络需要的一系列处理函数以及过程
    # 1. 对图片进行归一化
    #     1).按照指定尺寸缩放
    #     2).对缩放后的图片转换数据类型并且归一化
    # 2. 获得pnet的输入
    #     1).按照比例系数进行缩放
    # 3. 经过pnet网络，获得预测人脸框
    ######################################
    def pnet_processing(self, image):
        """封装pnet需要的函数"""
        
        # 传入图片，归一化
        image_ = self.__norm(image.copy())
        
        # 获得pnet的输入
        pnet_need_imgs = self.__get_pnet_need_imgs(image_)
        
        # 经过pnet网络，得到pnet预测的人脸框
        pnet_got_rects = self.__pnet.forward(pnet_need_imgs)
        
        if len(pnet_got_rects) == 0:
            
            self.print_messages("准备检测下一张图片")
            
            return [], False
        
        else:
            
            self.print_messages("pnet一共获得{}个预测人脸框".format(len(pnet_got_rects)))

        return pnet_got_rects, True
    
    def __norm(self, image):
        """对输入的图片作归一化"""
        
        # 颜色通道转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 对图片按用户指定的尺寸缩放
        if self.resize_type == 'with_loss':
            image = self.image_resize(image, 
                                      self.resize_shape)
        else:
            image = self.image_resize_padding(image, 
                                              self.resize_shape)
        
        # 对图片作归一化处理
        #image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5

        return image
        

    def image_resize(self,
                     image,
                     size):
        """图像缩放"""
        image = cv2.resize(image.copy(),
                           (size[0], size[1]))

        return image
  

    def image_resize_padding(self, 
                     image, 
                     size):
        """缩放函数
        
        Args:
            image: 待缩放的原始图像
            size: (tuple, (int, int)),缩放后的尺寸
        
        """
        
        width = image.shape[0] # 获得图像的宽
        height = image.shape[1]# 获得图像的高

        # 选择大的边作为resize后的边长
        side_length = image.shape[0] if width \
                                      > height \
                                     else height

        mask = self.mask_template((side_length, side_length))

        mask[0:width, 0:height] = image # 获取padding后的图像
        
        image = self.image_resize(mask, size)

        return image

        
    def mask_template(self, shape):
        """图片掩码模板
        
        根据用户输入resize图片的尺寸，
        制作模板，方便获取不同大小的pnet
        图片的需求
        
        """
        sss = np.zeros([shape[0], shape[1]],dtype=np.uint8)
        sss = cv2.cvtColor(sss, cv2.COLOR_GRAY2RGB)
        sss = (sss-127.5)/127.5 
        
        return sss
    
    ##############################################
    # 按照比例系数获得pnet的输入
    ##############################################
    def __get_pnet_need_imgs(self, image):
        """获得pnet输入需要的一系列图片
        
        通过scales对原始图片进行缩放，被
        缩放的图片填充回原图大小，打包返回
        
        Args:
            image: 需要被检测人脸的图像
            
        Returns：
            np.array((image1, image2, image3 ...))
            shape = (n, x, x, 3, np.array)
        
        """
        
        image_width = image.shape[0]
        image_height = image.shape[1]
        image_list = []
        
        for scale in self.__scales:
            
            sss_ = self.mask_template(self.resize_shape)
        
            width = int(scale*image_width)
            height = int(scale*image_height)
            size = (width, height)
            img_tmp = self.image_resize(image.copy(), size) 
                                        
            
            sss_[0:width, 0:height] = img_tmp

            image_list.append(sss_)
            
            #how_img(sss_)
            
        return np.array(image_list)
    
    ##########################################
    # rnet网络处理
    # 1. 归一化原始图像
    # 2. 获得rnet网络的输入
    # 3. 经过网络得到输出
    ##########################################
    def rnet_processiong(self, rects, image):
        """封装rnet需要的函数"""
        
        # 传入图片，归一化
        image_ = self.__norm(image.copy())
        
        # 获取rnet需要的输入数据
        rnet_need_imgs = self.get_rnet_need_imgs(image_, 
                                                 rects)
        
        # 经过网络获得预测结果
        rnet_got_rects = self.__rnet.forward(rnet_need_imgs, 
                                             rects,
                                             image_)
        
        if len(rnet_got_rects) == 0:
            
            self.print_messages("准备检测下一张图片")
            
            return [], False
        
        else:
            
            self.print_messages("rnet一共获得{}个预测人脸框".format(len(rnet_got_rects)))

        return rnet_got_rects, True        
    
    
    def get_rnet_need_imgs(self, image, rects):
        """获得一系列rnet需要的格式的图片

        Args:
            rects: 经过pnet处理得到的所有矩形框

        """

        rnet_need_imgs = self.__get_net_need_imgs(rects, 
                                                  image, 
                                                  'rnet')

        return np.array(rnet_need_imgs)


    def __get_net_need_imgs(self, rects, image, net_type='rnet'):
        """获取输入网络图像的通用方法

        判断rects的长度，如果为0，则返回空list

        rects的x1，y1不能大于图像长或者宽

        Args：
            image: 归一化后的原始图像
            rects: 经过网络处理后得到的矩形框
            net_type: 网络的类型，pnet或者rnet，(string)

        """

        need_imgs = []
        
        for rect in rects:

            tmp_roi = image.copy()[int(rect[1]): int(rect[3]), \
                                int(rect[0]): int(rect[2])]

            if net_type == 'rnet':

                tmp_roi = cv2.resize(tmp_roi, (24, 24))

                need_imgs.append(tmp_roi)

            elif net_type == 'onet':

                tmp_roi = cv2.resize(tmp_roi, (48, 48))

                need_imgs.append(tmp_roi)

        return need_imgs
    
    ##########################################
    # onet网络处理
    # 1. 归一化原始图像
    # 2. 获得onet网络的输入
    # 3. 经过网络得到输出
    ##########################################
    def onet_processing(self, rects, image):
        """封装rnet需要的函数"""
        
        # 传入图片，归一化
        image_ = self.__norm(image.copy())
        
        # 获取onet需要的输入数据
        onet_need_imgs = self.get_onet_need_imgs(image_, 
                                                 rects)
        
        # 经过网络获得预测结果
        onet_got_rects, landmarks = self.__onet.forward(onet_need_imgs, 
                                             rects,
                                             image_)
        
        if len(onet_got_rects) == 0:
            
            self.print_messages("准备检测下一张图片")
            
            return [], [], False
        
        else:
            
            self.print_messages("onet一共获得{}个预测人脸框".format(len(onet_got_rects)))

        return onet_got_rects, landmarks, True 
    
    ##########################################
    # 获取Onet输入的数据
    ##########################################


    def get_onet_need_imgs(self, image, rects):
        """获得一系列onet需要的格式的图片

        Args:
            rects: 经过rnet处理得到的所有矩形框

        """

        onet_need_imgs = self.__get_net_need_imgs(rects,
                                                  image,
                                                  'onet')

        return np.array(onet_need_imgs)
