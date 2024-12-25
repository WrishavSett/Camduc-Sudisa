from paintai.models import Camera

def get_total_camera_count_fn():
    total_count = Camera.objects.filter(isdeleted=False).count()
    return total_count

def activate_camera_fn(servicename,boolval):
    camera = Camera.objects.get(servicename=servicename, isdeleted=False)
    camera.isactivate = boolval
    camera.save()


def get_active_non_deleted_camera_count_fn():
    return Camera.objects.filter(isactivate=True, isdeleted=False).count()