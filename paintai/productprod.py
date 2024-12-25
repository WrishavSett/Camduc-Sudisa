# views.py or utils.py
from .models import ProductProduction, Camera, Product
from datetime import timedelta
from django.utils import timezone
# from django.db.models import Sum
from django.db.models import Q, Sum

# Initialize the in-memory cache
product_cache = {}
def get_product_id_from_cache_or_db(item_name: str):
    # Check if the item name is already in the cache
    if item_name in product_cache:
        print("Product Cache hit")
        return product_cache[item_name]
    
    # If not in cache, query the database
    try:
        item = Product.objects.get(name=item_name,isdeleted=False)
        # Store the result in the cache
        product_cache[item_name] = item
        print("Product Cache miss - Querying database")
        return item
    except Product.DoesNotExist:
        print("Product not found in the database")
        return None
    
camera_cache = {}
def get_camera_id_from_cache_or_db(id):
    # Check if the item name is already in the cache
    if id in camera_cache:
        print("Camera Cache hit")
        return camera_cache[id]
    
    # If not in cache, query the database
    try:
        item = Camera.objects.get(pk=id,isdeleted=False)
        # Store the result in the cache
        camera_cache[id] = item
        print("Camera Cache miss - Querying database")
        return item
    except Camera.DoesNotExist:
        print("Camera not found in the database")
        return None


# Create a new ProductProduction record
def create_product_production(camera_id, product_name, starttime, endtime, count=0):
    try:
        # Fetch Camera and Product instances
        print(camera_id,product_name)
        cameraid = get_camera_id_from_cache_or_db(camera_id) #Camera.objects.get(id=camera_id)
        productid = get_product_id_from_cache_or_db(product_name) #Product.objects.get(id=product_id)

        # Create ProductProduction instance
        production = ProductProduction(
            cameraid=cameraid,
            productid=productid,
            starttime=starttime,
            endtime=endtime,
            count=count
        )
        production.save()
        return production
    except Camera.DoesNotExist:
        print(f"Camera with ID {camera_id} not found.")
    except Product.DoesNotExist:
        print(f"Product with ID {product_name} not found.")
    except Exception as e:
        print(f"Error creating ProductProduction: {e}")
        return None

# Read ProductProduction by ID
def get_product_production_by_id(production_id):
    try:
        return ProductProduction.objects.get(id=production_id)
    except ProductProduction.DoesNotExist:
        print(f"ProductProduction with ID {production_id} not found.")
        return None

# Read all ProductProduction records (with optional filtering)
def get_all_product_productions_fn():
    return ProductProduction.objects.all()

def get_product_productions_by_camera(camera_id):
    return ProductProduction.objects.filter(cameraid=camera_id)

def get_product_productions_by_product(product_id):
    return ProductProduction.objects.filter(productid=product_id)

# Update ProductProduction
def update_product_production(production_id, starttime=None, endtime=None, count=None):
    try:
        production = ProductProduction.objects.get(id=production_id)
        if starttime:
            production.starttime = starttime
        if endtime:
            production.endtime = endtime
        if count is not None:
            production.count = count
        production.save()
        return production
    except ProductProduction.DoesNotExist:
        print(f"ProductProduction with ID {production_id} not found.")
        return None

# Delete ProductProduction
def delete_product_production(production_id):
    try:
        production = ProductProduction.objects.get(id=production_id)
        production.delete()
        print(f"ProductProduction with ID {production_id} deleted successfully.")
        return True
    except ProductProduction.DoesNotExist:
        print(f"ProductProduction with ID {production_id} not found.")
        return False


def get_product_counts(start_date, end_date):
    results = (
        ProductProduction.objects
        .filter(starttime__gte=start_date, endtime__lte=end_date)  # Filter by date range
        .values('cameraid','productid','productid__name')  # Group by productid
        .annotate(total_count=Sum('count'))  # Sum up the count field
    )
    print(results)
    return results



def get_product_counts_bycamproid(start_date, end_date, camera_id=-1, product_id=-1):
    # Build the filter criteria dynamically
    filters = Q(starttime__gte=start_date, endtime__lte=end_date)
    if camera_id != -1:
        filters &= Q(cameraid=camera_id)
    if product_id != -1:
        filters &= Q(productid=product_id)
    
    results = (
        ProductProduction.objects
        .filter(filters)  # Apply the dynamic filters
        .values('cameraid', 'productid', 'productid__name')  # Group by productid
        .annotate(total_count=Sum('count'))  # Sum up the count field
    )
    print(results)
    return results





def get_productProduction_top_five_last_day():
    last_24_hours = timezone.now() - timedelta(hours=24)
    
    # Query to get the top 5 products based on count in the last 24 hours
    top_selling_products = (ProductProduction.objects.filter(starttime__gte=last_24_hours)
                            .values('productid','productid__name')
                            .annotate(total_count=Sum('count'))
                            .order_by('-total_count')[:5])
    return top_selling_products