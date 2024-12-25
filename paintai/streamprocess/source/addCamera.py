import subprocess
import subprocess
import os
from django.conf import settings
from paintai.models import Camera
import time
# from elevate import elevate

nssm_path = settings.NSSMPATH # Path to NSSM executable

def create_copy_of_pyfile(original_file,new_file):
    try:
        with open(original_file, 'rb') as original:
            with open(new_file, 'wb') as new:
                new.write(original.read())
        print(f"File '{original_file}' duplicated as '{new_file}' successfully.")
    except FileNotFoundError:
        print(f"Error: File '{original_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def edit_python_file(file_path, old_content, new_content):
    try:
        # Read the content of the file
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Replace the old content with the new content
        modified_content = file_content.replace(old_content, new_content)

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(modified_content)

        print(f"File '{file_path}' edited successfully.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# def create_windows_service():
#     current_script_path = os.path.abspath(__file__)
#     parent_directory = os.path.dirname(os.path.dirname(current_script_path))
#     service_script = os.path.join(parent_directory, "addCamera.ps1")

#     # Paths
#     # service_script = r"C:\MyService\create_service.ps1"

#     # Ensure PowerShell execution policy allows scripts to run
#     try:
#         subprocess.run(["powershell", "Set-ExecutionPolicy", "RemoteSigned", "-Scope", "CurrentUser", "-Force"], check=True)
#         subprocess.run(["powershell", "-File", service_script], check=True)

#         subprocess.run(["powershell", "Start-Service", "-Name", "PythonService"], check=True)

#         print("Service created and started successfully.")
#     except Exception as e:
#         print(f'Error: {e}')



def create_nssm_service(service_name, executable_path, args, display_name, appdir):
    global nssm_path
    # elevate()
    # Construct the command to create the service
    install_command = [nssm_path, 'install', service_name,executable_path,args]
    set_appdir_command = [nssm_path, 'set', service_name,'AppDirectory', appdir]
    set_app_command = [nssm_path, 'set', service_name,'Application', executable_path]
    set_appargs_command = [nssm_path, 'set', service_name,'AppParameters', args]
    set_display_command = [nssm_path, 'set', service_name,'DisplayName', display_name]
    set_ffmpeg_env_command = [nssm_path, 'set', service_name,'AppEnvironmentExtra', f"PATH={settings.FFMPEGDIR}"]
    remove_command =  [nssm_path,'remove', service_name,'confirm']
    try:

        subprocess.run(install_command, shell=True)
        time.sleep(1)
        subprocess.run(set_app_command, shell=True)
        time.sleep(1)
        subprocess.run(set_appargs_command, shell=True)
        time.sleep(1)
        subprocess.run(set_ffmpeg_env_command, shell=True)
        time.sleep(1)        
        subprocess.run(set_display_command, shell=True)
        time.sleep(1)
        subprocess.run(set_appdir_command, shell=True)
        print(set_appdir_command)
        time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')
        subprocess.run(remove_command, shell=True)
        return
    print("Successfully Created Services")


def delete_nssm_service(id):
    global nssm_path
    camera = Camera.objects.get(pk=id)
    print(camera)
    new_camera_name = camera.name
    service_name = camera.servicename
    current_script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(current_script_path))
    new_file_paths = os.path.join(parent_directory,"cameras",f"{new_camera_name}.py")
    remove_command =  [nssm_path,'remove', service_name,'confirm']
    try:
        subprocess.run(remove_command, shell=True)
        os.remove(new_file_paths)
        camera.delete()
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')
        return
    print(f"Successfully Deleted Services {service_name}")

def start_nssm_service(service_name):
    global nssm_path
    start_command =  [nssm_path,'start', service_name]
    try:
        subprocess.run(start_command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')
        return
    print(f"Successfully Started Services {service_name}")

def stop_nssm_service(service_name):
    global nssm_path
    stop_command =  [nssm_path,'stop', service_name]
    try:
        subprocess.run(stop_command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')
        return
    print(f"Successfully Stopped Services {service_name}")

def create_new_camera_add_service(new_camera_name,new_camera_ip):
    current_script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(current_script_path))
    old_file_path = os.path.join(parent_directory,"source" ,'cameraSource.py')
    old_content = 'localhost'
    old_id = "2356"
    old_log = "D:"
    old_camera_name = "Camera110"
    new_log = os.path.join(settings.BASE_DIR,"paintai","logs")
    new_file_paths = [os.path.join(parent_directory,"cameras",f"{new_camera_name}.py")]
    new_contents = [new_camera_ip]
    for i,new_file_path in enumerate(new_file_paths):
        try:
            create_copy_of_pyfile(old_file_path,new_file_path)
            edit_python_file(new_file_path, old_content, new_contents[i])
            edit_python_file(new_file_path, old_log, new_log)
            edit_python_file(new_file_path, old_camera_name, new_camera_name)
            camera_extension = new_contents[i].split(".")[-1]
            create_nssm_service(
                service_name=f'SudisaPaintAI{camera_extension}',
                executable_path=settings.PYTHONVENVPATH,
                args=f"{new_camera_name}.py",
                appdir=settings.APPDIR,
                display_name=f"SudisaPaintAI{camera_extension}")
            camera = Camera(
                name=new_camera_name,
                ipaddr=new_camera_ip,
                servicename=f'SudisaPaintAI{camera_extension}'
            )
            camera.save()
            edit_python_file(new_file_path, old_id, str(camera.id))
        except Exception as e:
            print(str(e))

def get_camera_list_from_db():
    queryset = Camera.objects.all().values()
    data = list(queryset)
    return data

# if __name__=="__main__":
#     old_file_path = 'dummy.py'
#     old_content = '192.168.1.1'  
#     new_file_paths = ['newdummy1.py','newdummy2.py','newdummy3.py']
#     new_contents = ['192.168.1.102','192.168.1.201','192.168.1.303']
#     for i,new_file_path in enumerate(new_file_paths):
#         create_copy_of_pyfile(old_file_path,new_file_path)
#         edit_python_file(new_file_path, old_content, new_contents[i])
#         camera_extension = new_contents[i].split(".")[-1]
#         create_nssm_service(
#             service_name=f'SudisaAI{camera_extension}',
#             executable_path=settings.PYTHONVENVPATH,
#             args=new_file_path,
#             appdir= settings.APPDIR,
#             display_name=f"SudisaAI{camera_extension}")