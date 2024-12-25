from paintai.models import Product

def create_product_fn(name,yoloid):
    product = Product.objects.create(name=name,yoloid=yoloid)
    return product

def get_product_by_id_fn(product_id):
    product = Product.objects.get(pk=product_id, isdeleted=False)
    return product

def get_product_by_name_fn(product_name):
    product = Product.objects.get(name=product_name, isdeleted=False)
    return product

def get_all_products_fn():
    return Product.objects.filter(isdeleted=False)

def update_product_fn(product_id, new_name):
    product = Product.objects.get(pk=product_id, isdeleted=False)
    product.name = new_name
    product.save()
    return product

def soft_delete_product_fn(product_id):
    product = Product.objects.get(pk=product_id, isdeleted=False)
    product.isdeleted = True
    product.save()
    return True

def get_total_product_count_fn():
    total_count = Product.objects.filter(isdeleted=False).count()
    return total_count
