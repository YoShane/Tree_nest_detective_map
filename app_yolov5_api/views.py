from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, Http404

import os
import time
import urllib.request
from io import BytesIO
import base64
from pathlib import Path
from datetime import datetime


# YOLO影像辨識
from PIL import Image, ExifTags
import cv2
import numpy as np
import re
import platform
import shutil

import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.datasets import letterbox as original_size_to_letterbox
from tensorflow.keras.preprocessing.image import load_img, array_to_img

from my_yolov5_detect import yolo_detect_nest
from my_yolov5_detect import yolo_detect_coco

# 地圖標記
from app_yolov5_api.models import Pois_map
import simplekml


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device('cpu')


print("load models ...")

# 載入yolov5訓練好的模型(載入一次)
weights = './trained_models/best.pt'
#weights = './trained_models/yolov5m.pt'


model_yolo = attempt_load(weights, map_location=device)

# 獲取類別名字
names = model_yolo.names

# 設置畫框的顏色
rnd = np.random.RandomState(123)
colors = [[rnd.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


@csrf_exempt
def api_detect_upload(request):

    # 上傳過來的檔案存放於記憶體中
    # <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>
    imgBin = request.FILES["upload_image"]

    fileName = 'img_'+str(int(round(time.time() * 1000)))+'.jpg'
    # 若要將上傳的圖檔img_bin存檔至media目錄
    fs = FileSystemStorage()
    filePath = fs.save('app_detect/static/img/upload_img/'+fileName, imgBin)

    # Image.open()可以讀取InMemoryUploadedFile影像檔案
    # 也可以用ski image 讀取(參考影像AI上線:ResNet50專案)
    img_pil = Image.open(filePath)
    # print(type(img_pil)) # 印出影像格式看看

    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        print('Originally, this picture was '+img_pil.mode+' mode.')  # 印出影像模式

    # <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>

    # 手機直立拍照照片會旋轉，必須轉正才能正確偵測人臉
    # 判斷圖片是否有exif信息
    # 注意:有些照片有exif資訊，但是為None，需要判斷此情況，否則回報錯!
    if hasattr(img_pil, '_getexif') and img_pil._getexif() != None:
        try:
            dict_exif = img_pil._getexif()  # 獲得exif訊息
            # dict_exif=dict(img._getexif().items())

            if dict_exif[274] == 3:
                img_pil = img_pil.rotate(180, expand=True)
            elif dict_exif[274] == 6:
                img_pil = img_pil.rotate(270, expand=True)
            elif dict_exif[274] == 8:
                img_pil = img_pil.rotate(90, expand=True)
        except:
            print('判斷照片旋轉失敗!原因待查!')

    # PIL格式轉成numpy array
    img_origin = np.array(img_pil)

    img_result, obj_info = yolo_detect_nest(
        img_origin, model_yolo)  # 呼叫物件偵測

    # 在terminal印出偵測後的結果來瞧瞧，可以判斷一下答案對不對
    print(obj_info)

    # 準備要回傳到前端的資訊字典格式
    response = {}
    response['obj_info'] = obj_info
    response['time'] = datetime.now().strftime("%Y-%m-%d %H:%M")


    if obj_info == None:
        # if len(obj_info)==0:
        response['status'] = "Can't found any object."
        # 傳回上傳的圖片
        response['img_result'] = toBase64_Pic(img_pil,False)
        return JsonResponse(response)

    else:
         # objs_info本身是dict，此處增加一個'img_result'鍵值，以夾帶編碼後的影像到前端網頁
        response['img_result'] = toBase64_Pic(img_result,True)

        picLat = getPicLocation(filePath) #取得圖片XMP (拍攝位置)

        if(picLat != None):
            findSameLat = False

            for poi in Pois_map.objects.all():
                if(poi.point == picLat):
                    findSameLat = True
                    break

            if findSameLat:
                response['status'] = "This picture was detected."
                return JsonResponse(response)
            else:
                Pois_map.objects.create(title=obj_info[0].get("obj_name"),point=picLat,description="",picture=fileName )

                response['status'] = "Done."
                return JsonResponse(response)
        else:
            response['status'] = "Picture has not location information."
            return JsonResponse(response)

@csrf_exempt
def api_create_kml(request):

    kml = simplekml.Kml()
    for poi in Pois_map.objects.all():
        point = poi.point.replace(" ", "").split(",")
        plon = float(point[1])
        plat = float(point[0])

        des = poi.description + "<br>建立點：" + \
            poi.created_at.strftime("%Y-%m-%d")

        pnt = kml.newpoint(name=poi.title, description=des,
                           coords=[(plon, plat)])  # lon, lat optional height

        if(poi.title == "bird-nest"):
            pnt.style.iconstyle.icon.href = 'static/img/bird.png'
        elif(poi.title == "ant-nest"):
            pnt.style.iconstyle.icon.href = 'static/img/ant.png'
        elif(poi.title == "hornet-nest"):
            pnt.style.iconstyle.icon.href = 'static/img/bee.png'

    kml.save("pois.kml")

    file_path = "pois.kml"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            fileStr = fh.read()
            fileStr = fileStr.replace(
                bytes("<name>", encoding='utf8'), bytes("<title>", encoding='utf8'))
            fileStr = fileStr.replace(
                bytes("</name>", encoding='utf8'), bytes("</title>", encoding='utf8'))
            response = HttpResponse(
                fileStr, content_type="application/vnd.google-earth.kml+xml")
            response['Content-Disposition'] = 'inline; filename=' + \
                os.path.basename(file_path)
            return response
    raise Http404


def getPicLocation(img_path):
    with open(img_path, "rb") as fin:
        imgAsString = str(fin.read())

    xmp_start = imgAsString.find('<Iptc4xmpExt:Sublocation>')
    xmp_end = imgAsString.find('</Iptc4xmpExt:Sublocation>')
    if xmp_start != xmp_end:
        xmpString = imgAsString[xmp_start+25:xmp_end]
    else:
        xmpString = None

    return(xmpString)

def toBase64_Pic(img,isNumpy):

    output_buffer = BytesIO()  # 產生一個二進位檔案的buffer

    if isNumpy:
        # 回傳的img有畫物件框框(PIL格式)，要送到前端給網頁顯示，必須要使用特別的以base64編碼過字串
        # 讀取影像檔案，再轉成base64編碼(string格式)的寫法:
        img = array_to_img(img)  # 圖片轉回PIL格式
        img.save(output_buffer, format='PNG')  # 將img影像存到該二進位檔案的buffer
        byte_data = output_buffer.getvalue()  # 拿出該buffer的二進位格式資料
        # 編碼成base64再decode()轉碼成文字格式
        img_base64 = base64.b64encode(byte_data).decode()
    else:
        img.save(output_buffer, format='PNG')  # 將img影像存到該二進位檔案的buffer
        byte_data = output_buffer.getvalue()  # 拿出該buffer的二進位格式資料
        # 編碼成base64再decode()轉碼成文字格式
        img_base64 = base64.b64encode(byte_data).decode()

    return img_base64


# 啟動時測試一下偵測是否正常
img_path = './test_images/1.jpg'

img_origin = load_img(img_path)
img_origin = np.array(img_origin)
img_result, obj_info = yolo_detect_nest(
    img_origin, model_yolo)
print(obj_info)
