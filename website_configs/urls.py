"""website_configs URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from app_yolov5_api import views

from app_detect import views as views_detect

urlpatterns = [
    path('',views_detect.home, name='home'),
    path('map',views_detect.get_map, name='map'),
    path('list',views_detect.list_page, name='list'),
    path('api/detect', views.api_detect_upload, name='api_detect'),
    path('api/get_pois', views.api_create_kml, name='api_pois'),
]
