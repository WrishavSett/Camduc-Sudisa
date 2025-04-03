import cv2
from ultralytics import solutions

def count_objects_in_region(rtsp_url, output_video_path, model_path, target_width=1280, target_height=720, original_width=2560, original_height=1440):
    """Count objects in a specific region within a video stream."""
    cap = cv2.VideoCapture(rtsp_url)
    assert cap.isOpened(), "Error opening RTSP stream"
    
    # Get video properties (width, height, fps)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    # Resize video output dimensions to fit in the window
    if w > target_width or h > target_height:
        w, h = target_width, target_height
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Define the region points for object counting
    region_points = [(200, 400), (2360, 400), (2360, 1000), (200, 1000)]
    
    # Calculate scaling factors for width and height
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # Scale the region points according to the new resolution
    scaled_region_points = [(int(x * scale_x), int(y * scale_y)) for x, y in region_points]
    
    counter = solutions.ObjectCounter(show=False, region=scaled_region_points, model=model_path, verbose=False)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Failed to grab a frame or the RTSP stream is finished.")
            break
        
        # print(im0)
        # Resize the frame to fit within the desired resolution
        im0 = cv2.resize(im0, (target_width, target_height))  # Resize the frame

        results = counter.count(im0)  # Count objects in the frame
        # Write the processed frame to the output video
        video_writer.write(results)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  # Close all opened windows

# Example RTSP URL
# rtsp_url = "rtsp://admin:hikvision%23123@192.168.3.66:554/Streaming/channels/101"
rtsp_url = "D:/Camduc-Sudisa/paintai/videos/20250324_113101.mp4"
count_objects_in_region(rtsp_url, "C:/Camduc-Sudisa/output_video.avi", "D:/Camduc-Sudisa/Iter3.1s.pt", 1280, 720)
