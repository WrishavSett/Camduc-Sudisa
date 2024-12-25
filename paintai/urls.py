from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("home", views.hometemplate, name="hometemplate"),
    path("products", views.producttemplate, name="producttemplate"),
    path("cameras", views.cameratemplate, name="cameratemplate"),
    path("production", views.productiontemplate, name="cameratemplate"),


    path("addcamera",views.addCamera, name="addCamera"),
    path("getallcameras",views.getAllCameras, name="getallcameras"),
    path("startService",views.startService, name="startService"),
    path("stopService",views.stopService, name="stopService"),

    path('createproduct', views.create_product, name='create_product'),
    path('getproduct/<int:product_id>', views.get_product, name='get_product'),
    path('getproductbyname/<str:product_name>', views.get_product_by_name, name='get_product_by_name'),
    path('getallproduct', views.get_all_products, name='get_all_products'),


    path('getproductscount', views.get_total_product_count, name='get_total_product_count'),
    path('getacamerascount', views.get_total_camera_count, name='get_total_camera_count'),
    path('getactivecamerascount', views.get_total_active_camera_count, name='get_total_active_camera_count'),

    
    path('deleteproduct/<int:product_id>', views.soft_delete_product, name='soft_delete_product'),

    path('getallproduction', views.get_all_production, name='get_all_production'),
    path('getproductionbydate', views.get_production_by_date, name='get_production_by_date'),
    path('getproductionbyfilter', views.get_production_by_filter, name='get_production_by_filter'),
    path('gettopproduction', views.get_production_top5, name='get_production_top5'),
    path('download-csv/', views.get_production_by_date_download_csv, name='download_csv'),
    path('createproduction', views.create_production, name='get_all_production'),


    path('getcampayload', views.getCameraPayload, name='getCameraPayload'),


]