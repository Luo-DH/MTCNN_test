## MTCNN

### 使用说明

- 引入相关头文件

```python
from src.MTCNN import MTCNN
```

- 构建mtcnn对象

```python
mtcnn = MTCNN()
```

> MTCNN类几个重要参数
>
> - max_face: 是否检测最大人脸，如果max_face==False, 则会检测图像中所有人脸
> - save_face: 是否保存检测到的人脸
> - save_dirt: 保存人脸的目录(如果设定了save_face=True), 则必须设置该参数，默认路径为"./output"
> - detect_hat: 是否检测安全帽佩戴，True/False

```python
mtcnn = MTCNN(threshold=[0.5, 0.6, 0.7], # 置信度阈值
              					  nms_threshold=[0.7, 0.7, 0.7], # nms阈值
             				 	  weight_paths=['./weight_path/pnet.h5', # 权重文件路径
                            										'./weight_path/rnet.h5',
                           								 			'./weight_path/onet.h5'],
                                  max_face=False, # 是否检测最大人脸
                                  save_face=True, # 是否保存检测到的人脸
                                  save_dirt="./output", # 保存人脸的路径
                                  print_time=True, # 是否打印时间信息
                                  print_message=True, # 是否打印辅助信息
                                  detect_hat=True, # 是否检测安全帽佩戴
                                  resize_type='without_loss', # resize图片类型
                                  padding_type='mask' # 填充图片类型
              )
```



- 创建好mtcnn对象后，只需要调用其detect_face方法，就可以完成人脸检测

```python
mtcnn.detect_face(image)
```

> 传入的图片最好是opencv读取所得

- 调用detect_face方法后，有三个返回值

  >- 如果检测到人脸：
  >
  >   - rects: (list, float): 返回检测到矩形框的左上角和右下角坐标 [[x1,y1,x2,y2]]
  >
  >     ```python
  >     for rect in rects:
  >     
  >         img_ = cv2.rectangle(image.copy(), 
  >                              (int(rect[0]), int(rect[1])),
  >                              (int(rect[2]), int(rect[3])),
  >                              (0, 255, 0), 4)
  >         cv2.imshow("img_", img_)
  >         cv2.waitKey(0)
  >         cv2.destroyAllWindows()
  >     ```
  >
  >  - landmarks: 返回左眼，右眼，鼻子，左嘴，右嘴的坐标，用于人脸矫正使用
  >
  >  - ret: True 成功检测到人脸
  >
  >- 如果没有检测到人脸
  >
  >   - 返回 [], [], False

  