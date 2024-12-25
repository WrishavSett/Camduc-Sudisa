from django.db import models
from django.utils import timezone
# Create your models here.
class Camera(models.Model):
    name = models.CharField(max_length=50, unique=True)
    servicename = models.CharField(max_length=50)
    ipaddr = models.GenericIPAddressField()
    isdeleted = models.BooleanField(default=False)
    createdon = models.DateTimeField(default=timezone.now)
    isactivate = models.BooleanField(default=False)


class Product(models.Model):
    name = models.CharField(max_length=100)
    yoloid = models.PositiveIntegerField(default=0)
    isdeleted = models.BooleanField(default=False)
    createdon = models.DateTimeField(default=timezone.now)

class ProductProduction(models.Model):
    cameraid = models.ForeignKey(Camera,on_delete=models.CASCADE)
    productid = models.ForeignKey(Product,on_delete=models.CASCADE)
    starttime = models.DateTimeField()
    endtime = models.DateTimeField()
    count = models.PositiveIntegerField(default=0)
