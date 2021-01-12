from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import requests
from PIL import Image

import matplotlib.pyplot as plt
import io

import cv2
import numpy as np
import base64
from app_yolov5_api.models import Pois_map


api_url = "http://127.0.0.1:8000/api/detect"

# Create your views here.
def get_map(request):
    return render(request, 'app_detect/map.html')

def list_page(request):
    pois = Pois_map.objects.all()
    response = { 'pois' : pois }

    if request.method == 'POST':
        id = int(request.POST.get('id',''))
        if(int(request.POST.get('delPoint','0')) == 1):
            Pois_map.objects.filter(pk=id).delete()
        else:
            Pois_map.objects.filter(pk=id).update(
                title=request.POST.get('title',''),description=request.POST.get('description',''),point=request.POST.get('location',''))
    
    return render(request, 'app_detect/list.html',response)

def home(request):
    if request.method == 'POST':

        img_bin = request.FILES["upload_image"]
        # 酬載 (payload)
        payload = {"upload_image": img_bin}

        # submit the request
        result = requests.post(api_url, files=payload)
        print(result)
        result = result.json()  # 得到的是request Response物件，須轉成json格式

        # 得到的影像檔案是base64編碼的字串，可以直接丟給前端顯示
        img_result_b64 = result['img_result']

        response = {
            'img_result_b64': img_result_b64,
            'obj_info': result['obj_info'],
            'status': result['status'],
            'time': result['time']
        }
        return render(request, 'app_detect/index.html', response)

    return render(request, 'app_detect/index.html')



def home_v2(request):

    if request.method == 'POST':

        img_bin = request.FILES["upload_image"]
        # 酬載 (payload)
        payload = {"upload_image": img_bin}

        # submit the request
        result = requests.post(api_url, files=payload)
        print(result)
        result = result.json()  # 得到的是request Response物件，須轉成json格式

        
        # 得到的影像檔案是base64編碼的字串，可以直接丟給前端顯示
        img_result_b64 = result['img_result']

        response ={
            'img_result_b64': img_result_b64,
            'obj_info': result['obj_info'],
            'status':result['status']
        }
        return render(request, 'app_detect/home.html', response)

    return render(request, "app_detect/home.html")


def home_v1(request):

    if request.method == 'POST':
        # # 上傳過來的檔案存放於記憶體中
        # # <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>
        # # 讀取前端網頁送過來的影像檔案InMemoryUploadedFile
        img_bin = request.FILES["upload_image"]

        # # 將上傳的圖檔img_bin存檔至media目錄
        # # 檔案存放路徑專案根目錄/media/img_uploaded.jpg
        # # 原有檔案不會被覆蓋  新檔名會使用原檔名加上隨機碼 多使用者沒問題
        # print(type(img_bin))
        fs = FileSystemStorage()
        file_path = fs.save('img_uploaded.jpg', img_bin)
        # print(file_path)
        file_path = fs.url(file_path)
        response = {
            'info_from_server': "上傳存檔成功!",
            'uploaded_img': file_path,
        }
        return render(request, 'app_detect/home.html', response)

    return render(request, "app_detect/home.html")


def home_v0(request):
    return render(request, "app_detect/home.html")
