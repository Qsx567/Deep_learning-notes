"""
    根据yolov5训练好的模型进行推理,使用的是PyTorch Hub加载模型
"""
import cv2
import numpy as np
import torch
import time

class PPE_detector:

    def __init__(self):
        # 加载模型
            # 第一个参数 -> yolov5的工程目录
            # 第二个参数 -> custom表示是自定义的
            # 第三个参数 -> 训练的模型
            # 第四个参数 -> local 表示本地加载
        self.model = torch.hub.load('../yolov5','custom',path='weights/ppe_yolo_n.pt',source='local') # local repo
        self.model.conf = 0.4 # NMS confidence threshold
        # model.iou = 0.45  # NMS IoU threshold

        # 获取视频流
        self.cap = cv2.VideoCapture(0)

        # 获取图像的w,h
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(self.frame_w,self.frame_h)

        # 标签
        labels =  ['person','vest','blue helmet','red helmet','white helmet','yellow helmet'] 

    
    def detect(self):
        
        # 获取视频流的每一帧
        while True:

            ret,frame = self.cap.read()

            # 画面翻转
            frame = cv2.flip(frame,1)

            # 将画面转化为RGB格式
            img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 计算推理时间
            start_time = time.time()
            # 执行推理过程
            results = self.model(img_cvt)
            # pd = results.pandas().xyxy[0]
            results_np = results.pandas().xyxy[0].to_numpy()

            # print(results_np)
            
            # person_list = pd[pd['name']=='person'].to_numpy()
            # vest_list = pd[pd['name']=='vest'].to_numpy()
            
            # print(person_list,vest_list)            


            end_time = time.time()
            fps_text = 1 / (end_time - start_time)

            cv2.putText(frame,'FPS: ' + str(round(fps_text,2)),(30,50),cv2.FONT_ITALIC,1,(0,255,0),2)

            # 绘制边界框
            for box in results_np:
                x_min,y_min,x_max,y_max = box[:4].astype('int') # 获取四个坐标，并转化为整数
                label_id = box[5]
                classes = box[6]

                if label_id == 0:
                    cv2.putText(frame,classes,(x_min+10,y_min-10),cv2.FONT_ITALIC,1,(0,255,0),1)
                    cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(255,0,0),5)
                else:
                    cv2.putText(frame,classes,(x_min+10,y_min-10),cv2.FONT_ITALIC,1,(0,255,0),1)
                    cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(255,255,0),5)
                
                cv2.putText(frame,classes,(x_min+10,y_min-10),cv2.FONT_ITALIC,1,(0,255,0),2)

            # 显示
            cv2.imshow("PPE_demo",frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# 实例化
ppe_detector = PPE_detector()
ppe_detector.detect()