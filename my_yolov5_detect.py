import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.datasets import letterbox as original_size_to_letterbox

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img


def yolo_detect_coco(img_origin, model_yolo, img_size = 416):

    # 獲取類別名字
    names = model_yolo.names

    # 設置畫框的顏色
    rnd = np.random.RandomState(123)
    colors = [[rnd.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    #### 準備yolo的輸入圖片
    """
    img_origin 原size圖片   用途:畫外框用
    img 進行resize+pad之後的圖片 格式為通道在前 (3, h, w) 用途:模型偵測用
    """
    # 將原圖依據原始長寬比率去縮放到要求的尺寸，不足的區域填上灰色
    img = original_size_to_letterbox(img_origin, new_shape=img_size)[0]

    # Reshape  格式轉成通道在前
    img = img.transpose(2, 0, 1)  # convert from 416x416x3 to 3x416x416
    # img = np.ascontiguousarray(img) # 讓array資料連續位置存放 運算比較快

    img = torch.from_numpy(img) # to torch tensor
    img = img.float()/255.0 # uint8 to fp16/32 --> 0-1
    img = img.unsqueeze(0) #reshape成為4個維度 (1,3,height,width) (神經網路的輸入)

    #### 偵測可能物件
    """
    前向傳播 返回pred的shape是(1, num_boxes, 5+num_class)
    h,w為傳入網路圖片的長和寬，注意dataset在檢測時使用了矩形推理，所以這裡h不一定等於w
    num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
    pred[..., 0:4]為預測框座標
    預測框座標為xywh(中心點+寬長)格式
    pred[..., 4]為objectness可信度
    pred[..., 5:-1]為分類結果
    """
    pred = model_yolo(img)[0]  # 只有一張圖 因此選取編號0即可

    
    ##### 過濾最可能的物件，依據信心值與iou閾值
    # Apply NMS 進行NMS
    """
    pred:前向傳播的輸出，是一個列表長度為batch_size
    conf_thres:置信度閾值
    iou_thres:iou閾值
    classes:是否只保留特定的類別
    agnostic:進行nms是否也去除不同類別之間的框
    輸出結果det為每個物件的資訊[x1,y1,x2,y2,conf,cls] 外框格式：xyxy(左上角右下角)
    """
    det = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0] # 只有一張圖 因此選取編號0即可

    # 若沒有物件返回
    if det is None:
        print("沒有偵測到物件")
        return [], None # 後面的步驟不要做了

    #####
    ##### 對每一個偵測到的物件做處理Process detections
    # 將det的xyxy尺寸調整為original size圖片的座標
    # Rescale boxes from img_size to original size
    # 調整預測框的座標，將img_size(經過resize+pad)的圖片的座標-->轉為original size圖片的座標
    # det_scaled內每個物件的資訊為[x1,y1,x2,y2,conf,cls]    
    # det_scaled --> tensor([[108.00000, 399.00000, 148.00000, 463.00000,   0.82306,   0.00000],
    #                        [263.00000, 393.00000, 307.00000, 472.00000,   0.79723,   0.00000]])
    det_scaled = det.clone().detach()
    det_scaled[:, :4] = scale_coords(img.shape[2:], det_scaled[:, :4], img_origin.shape).round()

    ####
    #### 針對每個偵測物件畫上外框
    # Process detections對每一個偵測到的物件做處理
    # 原始圖片img_origin的複製，用來畫偵測到物件外框
    img_result = img_origin.copy()

    ## 有多個物件被偵測到，一一處理之
    photo_obj_info=[]
    for obj_id, (*xyxy, conf, class_num) in enumerate(det_scaled):

        # 物件名稱外框等資訊
        x1,y1,x2,y2 = [int(val.tolist()) for val in xyxy]
        box = [x1,y1,x2,y2]
        # print(box)

        # 進行物件圖片切割
        #obj_img = img_origin[y1:y2, x1:x2]
        #print(obj_img.shape)
        #save_img('output/id_{}.jpg'.format(obj_id),obj_img)

        obj_info = {} #輸出dictionary
        obj_name = names[ int(class_num) ] # 物件名稱
        obj_info['obj_id'] = obj_id
        obj_info['obj_name'] = obj_name
        obj_info['confidence'] = round(float(conf),2)
        obj_info['box'] = box

        # photo object info加入每個物件資訊
        photo_obj_info.append(obj_info)
        # print(obj_info)

        # 在img_result(原圖大小)上畫物件外框
        label = "{},{}{:.2f}".format(obj_id, obj_name, conf) # 0,bus0.89
        # draw box on img_result
        plot_one_box(xyxy, img_result, label=label, color=colors[int(class_num)], line_thickness=2)
        #----------------
        # end of for each object

    #### Save resulted image (image with detections)
    #save_img("./img_detected.jpg", img_result)
    return img_result, photo_obj_info


# 這個函數很複雜，可以置放於 my_yolov5_detect.py 透過import使用
# from my_yolov5_detect import yolo_detect as yolo_detect2 
# 詳細分解動作請參看另給的Jupyter notebook的教材
def yolo_detect_nest(img_origin, model_yolo,  img_size = 416):

    # 獲取類別名字
    names = model_yolo.names

    # 設置畫框的顏色
    rnd = np.random.RandomState(123)
    colors = [[rnd.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    #### 輸入圖片size必須為32的倍數，若不是，會自動調整為32的倍數
    #img_size = check_img_size(img_size, s=model_yolo.stride.max())  # check img_size 

    #### 準備yolo的輸入圖片
    """
    img_origin 原size圖片   用途:畫外框用
    img 進行resize+pad之後的圖片 格式為通道在前 (3, h, w) 用途:模型偵測用
    """
    # 將原圖依據原始長寬比率去縮放到要求的尺寸，不足的區域填上灰色
    img = original_size_to_letterbox(img_origin, new_shape=img_size)[0]

    # Reshape  格式轉成通道在前
    img = img.transpose(2, 0, 1)  # convert from 416x416x3 to 3x416x416
    img = np.ascontiguousarray(img) # 讓array資料連續位置存放 運算比較快

    img = torch.from_numpy(img) # to torch tensor
    img = img.float()/255.0 # uint8 to fp16/32 --> 0-1
    img = img.unsqueeze(0) #reshape成為4個維度 (1,3,height,width) (神經網路的輸入)

    #### 偵測可能物件
    """
    前向傳播 返回pred的shape是(1, num_boxes, 5+num_class)
    h,w為傳入網路圖片的長和寬，注意dataset在檢測時使用了矩形推理，所以這裡h不一定等於w
    num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
    pred[..., 0:4]為預測框座標
    預測框座標為xywh(中心點+寬長)格式
    pred[..., 4]為objectness可信度
    pred[..., 5:-1]為分類結果
    """
    pred = model_yolo(img)[0]  # 只有一張圖 因此選取編號0即可

    
    ##### 過濾最可能的物件，依據信心值與iou閾值
    # Apply NMS 進行NMS
    """
    pred:前向傳播的輸出，是一個列表長度為batch_size
    conf_thres:置信度閾值
    iou_thres:iou閾值
    classes:是否只保留特定的類別
    agnostic:進行nms是否也去除不同類別之間的框
    輸出結果det為每個物件的資訊[x1,y1,x2,y2,conf,cls] 外框格式：xyxy(左上角右下角)
    """
    det = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0] # 只有一張圖 因此選取編號0即可

    # 若沒有物件被偵測到，返回
    if det is None:
        print("沒有偵測到物件")
        return None, None # 後面的步驟不要做了

    #####
    ##### 對每一個偵測到的物件做處理Process detections
    # 將det的xyxy尺寸調整為original size圖片的座標
    # Rescale boxes from img_size to original size
    # 調整預測框的座標，將img_size(經過resize+pad)的圖片的座標-->轉為original size圖片的座標
    # det_scaled內每個物件的資訊為[x1,y1,x2,y2,conf,cls]    
    # det_scaled --> tensor([[108.00000, 399.00000, 148.00000, 463.00000,   0.82306,   0.00000],
    #                        [263.00000, 393.00000, 307.00000, 472.00000,   0.79723,   0.00000]])
    det_scaled = det.clone().detach()
    det_scaled[:, :4] = scale_coords(img.shape[2:], det_scaled[:, :4], img_origin.shape).round()

    ####
    #### 針對每個偵測物件畫上外框
    # Process detections對每一個偵測到的物件做處理
    # 原始圖片img_origin的複製，用來畫偵測到物件外框
    #img_result = img_origin.copy()
    img_result = img_origin.copy()
    img_height, img_width = img_result.shape[0:2]
    

    ## 有多個物件被偵測到，一一處理之
    photo_obj_info=[]  #存放物件資訊的list
    for obj_id, (*xyxy, conf, class_num) in enumerate(det_scaled):
        # 物件名稱外框等資訊
        x1,y1,x2,y2 = [int(val.tolist()) for val in xyxy]
        #box = [x1,y1,x2,y2] # 未加大前的尺寸
        #print(box)              

        
        # 稍微把切臉的框框加大些:目的是將頭髮等性別的特徵納入，提高判別準確度
        hy = y2 - y1
        wx = x2 - x1
        ws = 0.2 #自訂:寬度加大比率
        hs = 0.2 #自訂:高度加大比率
        x1, y1, x2, y2 = max(0,int(x1-wx*ws)),max(0,int(y1-hy*hs)),  min(img_width,int(x2+wx*ws)), min(img_height,int(y2+hy*hs)) # 需考慮加大後超出圖片的情況

        box = (x1, y1, x2, y2) # 加大後的尺寸
        
        # 進行物件圖片切割crop object
        obj_img = img_origin[y1:y2, x1:x2]
        #print(obj_img.shape)
        # 物件圖片存檔
        # save_img('output/obj_id_{}.jpg'.format(obj_id), obj_img)
        
        # 物件資訊
        obj_info = {} #輸出dictionary
        obj_name = names[ int(class_num) ] # 物件名稱
        obj_info['obj_id'] = obj_id
        obj_info['obj_name'] = obj_name
        obj_info['confidence'] = round(float(conf),2)
        obj_info['box'] = box

        ### 若沒有進行分類任務 以下程式碼可以刪除----------------------
  

        #其他分類或功能
        
        #obj_info["age"] = {'age':age} 
        # print(obj_info)


        # photo object info加入每個物件資訊
        photo_obj_info.append(obj_info)
        

        # 在img_result(原圖大小)上畫物件外框
        box_label = "{},{}{:.2f}".format(obj_id, obj_name, conf) # 0,bus0.89
        #box_label = "{},{}{}%age{}".format(obj_id, gender_label, gender_proba, age) #加上文字資訊
        # draw box on img_result
        plot_one_box(xyxy, img_result, label=box_label, color=colors[int(class_num)], line_thickness=3)
        #----------------
        # end of for each object

    #### Save resulted image (image with detections)
    # save_img("./media/img_result.jpg", img_result)
    if img_result.shape[0] > 800 :
        img_result = original_size_to_letterbox(img_result, new_shape=800)[0]
    #print(img_result.shape)
    return img_result, photo_obj_info