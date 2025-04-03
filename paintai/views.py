from django.shortcuts import render
from .streamprocess.source.addCamera import create_new_camera_add_service,\
    delete_nssm_service,start_nssm_service,stop_nssm_service,get_camera_list_from_db
from .products import *
from .productprod import *
from .productprod import *
from .camera import *
import json
from django.http import HttpResponse,JsonResponse
from datetime import datetime
import time 
import csv
from django.shortcuts import render

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def get_total_camera_count(request):
    try:
        total_count = get_total_camera_count_fn()
        return JsonResponse({"total_count": total_count}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def get_total_product_count(request):
    try:
        total_count = get_total_product_count_fn()
        return JsonResponse({"total_count": total_count}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def get_total_active_camera_count(request):
    try:
        total_count = get_active_non_deleted_camera_count_fn()
        return JsonResponse({"total_count": total_count}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    

def addCamera(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            cameraname = data.get("cameraname")
            ipaddr = data.get("ipaddr")
            create_new_camera_add_service(cameraname,ipaddr)
            return JsonResponse({"message": "Service Created", "name": cameraname, "IP": ipaddr})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    elif request.method == "DELETE":
        try:
            data = json.loads(request.body)
            cameraid = data.get("cameraid")
            delete_nssm_service(cameraid)
            return JsonResponse({"message": "Camera Deleted", "name": cameraid})
        except Exception as e:
                return JsonResponse({"error": str(e)}, status=400)
    else:
        return HttpResponse("Please send a post request")


def getAllCameras(request):
    data = None
    try:
        data = get_camera_list_from_db()
        return JsonResponse(data, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


def startService(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            servicename = data.get("servicename")
            start_nssm_service(servicename)
            activate_camera_fn(servicename,True)
            return JsonResponse({"message": "Service Started", "name": servicename})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return HttpResponse("Please send a post request")
    

def stopService(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(data)
            servicename = data.get("servicename")
            stop_nssm_service(servicename)
            print("service Stopped")
            activate_camera_fn(servicename,False)
            print("DB Updated")
            return JsonResponse({"message": "Service Stopped", "name": servicename})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return HttpResponse("Please send a post request")
    

def create_product(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            name = data.get("name")
            yoloid = data.get("yoloid")
            if not name:
                return JsonResponse({"error": "Name is required"}, status=400)
            product = create_product_fn(name=name,yoloid=yoloid)
            return JsonResponse({"message": "Product created", "id": product.id, "name": product.name}, status=201)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Invalid request method"}, status=405)

def get_product(request, product_id):
    if request.method == "GET":
        try:
            product = get_product_by_id_fn(product_id)
            return JsonResponse({"id": product.id, "name": product.name, "createdon": product.createdon}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


def get_product_by_name(request, product_name):
    if request.method == "GET":
        try:
            product = get_product_by_name_fn(product_name)
            print(product)
            return JsonResponse({"id": product.id, "name": product.name, "createdon": product.createdon}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
        
def get_all_products(request):
    if request.method == "GET":
        try:
            products = get_all_products_fn()
            product_list = [{"id": p.id, "name": p.name, "createdon": p.createdon} for p in products]
            return JsonResponse({"products": product_list}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

def soft_delete_product(request, product_id):
    if request.method == "DELETE":
        try:
            product = soft_delete_product_fn(product_id)
            return JsonResponse({"message": "Product is deleted"},status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Invalid request method"}, status=405)



def get_all_production(request):
    if request.method == "GET":
        try:
            products = get_all_product_productions_fn()
            product_list = [{"id": p.id, "camera_id": p.cameraid.id,"productid": p.productid.id,"count": p.count, "starttime": p.starttime,"endtime": p.endtime} for p in products]
            return JsonResponse({"product_productions": product_list}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


def get_production_by_date(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            starttime = datetime.strptime(data.get("starttime"), "%Y-%m-%dT%H:%M:%S.%f")
            endtime = datetime.strptime(data.get("endtime"), "%Y-%m-%dT%H:%M:%S.%f")
            products = get_product_counts(starttime,endtime)
            product_list =  list(products)
            return JsonResponse({"product_productions": product_list}, status=200,safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

def get_production_by_filter(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            starttime = datetime.strptime(data.get("starttime"), "%Y-%m-%dT%H:%M:%S.%f")
            endtime = datetime.strptime(data.get("endtime"), "%Y-%m-%dT%H:%M:%S.%f")
            camera_id = data.get("cameraid", -1)
            product_id = data.get("productid", -1)
            products = get_product_counts_bycamproid(starttime, endtime, camera_id, product_id)
            product_list =  list(products)
            return JsonResponse({"product_productions": product_list}, status=200,safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)




def get_production_top5(request):
    if request.method == "GET":
        try:
            products = get_productProduction_top_five_last_day()
            product_list =  list(products)
            return JsonResponse({"product_productions": product_list}, status=200,safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

def get_production_by_date_download_csv(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            starttime = datetime.strptime(data.get("starttime"), "%Y-%m-%dT%H:%M:%S.%f")
            endtime = datetime.strptime(data.get("endtime"), "%Y-%m-%dT%H:%M:%S.%f")
            products = get_product_counts(starttime,endtime)
            product_list =  list(products)
            data = {"product_productions": product_list}
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="product_productions.csv"'
            writer = csv.writer(response)
            writer.writerow(['Camera ID', 'Product ID', 'Product Name', 'Total Count'])
            for production in data["product_productions"]:
                writer.writerow([
                    production["cameraid"],
                    production["productid"],
                    production["productid__name"],
                    production["total_count"],
                ])
            return response
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)



def create_production(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(data)
            return JsonResponse({"message": "Service Created"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return HttpResponse("Please send a post request")


def getCameraPayload(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(data)
            cameraid = data.pop('cameraid')
            starttime = datetime.strptime(data.pop('starttime'), "%Y-%m-%dT%H:%M:%S.%f")
            endtime = datetime.strptime(data.pop('endtime'), "%Y-%m-%dT%H:%M:%S.%f")
            for key,value in data.items():
                create_product_production(cameraid, key, starttime, endtime, sum(value.values()))
                time.sleep(1)
            return JsonResponse({"message": "Service Created"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return HttpResponse("Please send a post request")
    

def hometemplate(request):
    return render(request,'index.html')

def cameratemplate(request):
    return render(request,'camera.html')

def producttemplate(request):
    return render(request,'product.html')

def productiontemplate(request):
    return render(request,'production.html')
