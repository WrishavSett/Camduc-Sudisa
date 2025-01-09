import os
import cv2
import time
import ffmpeg
import logging
import threading
import requests
import numpy as np
from datetime import datetime
from ObjectCount import ObjectCounter
from queue import Queue

class CAMERAMODEL:
    def __init__(self):
        self.camera_config = {
            "camera": "PaintCam1",
            "camera_id": int("2"),
            "region_points": [(1250, 0), (1250, 1440)],
            "url": "http://localhost:8000/ai/getcampayload",
            "logdir": "D:/RohitDa/Camduc/paintai/logs",
            "videodir": "D:/RohitDa/Camduc/paintai/videos",
            "processed_logdir": "D:/RohitDa/Camduc/paintai/processed_logs",
            "rtsp_url": "rtsp://localhost:18554/mystream"
        }

        self.counter = ObjectCounter(
            show=False,
            region=self.camera_config['region_points'],
            model="D:/RohitDa/Camduc/paintai/streamprocess/cameras/SudisaItem12K.pt",
            show_in=False,
            show_out=False,
            verbose=False,
        )

        os.makedirs(self.camera_config['logdir'], exist_ok=True)
        os.makedirs(self.camera_config['videodir'], exist_ok=True)
        os.makedirs(self.camera_config['processed_logdir'], exist_ok=True)

        self.logger = logging.getLogger(self.camera_config['camera'])
        self.logger.setLevel(logging.DEBUG)
        time_rotation = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(self.camera_config['logdir'], self.camera_config['camera'] + '.log'),
            when='W5',
            backupCount=1
        )
        logFormat = logging.Formatter(
            '%(asctime)s - %(name)s - %(threadName)s - %(funcName)s - %(levelname)s - %(message)s'
        )
        time_rotation.setFormatter(logFormat)
        time_rotation.setLevel(logging.DEBUG)
        self.logger.addHandler(time_rotation)

        self.video_queue = Queue()  # Queue to handle videos for object counting
        self.args = {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay"
        }
        try:
            self.logger.info("Probing RTSP Stream for camera: {}".format(self.camera_config['rtsp_url']))
            probe = ffmpeg.probe(self.camera_config['rtsp_url'], **self.args)
            cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
            self.logger.info("fps: {}".format(cap_info['r_frame_rate']))
            self.width = cap_info['width']
            self.height = cap_info['height']
            up, down = str(cap_info['r_frame_rate']).split('/')
            self.fps = eval(up) / eval(down)
            self.logger.info(f"fps: {self.fps} and height: {self.height} and width: {self.width}")
        except Exception as e:
            self.logger.error("Failed to Probe RTSP Stream")
            self.logger.error(e)
            raise e

    def save_video_stream(self):
        """Thread for saving the RTSP stream into 10-minute videos."""
        while True:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_path = os.path.join(self.camera_config['videodir'], f"{timestamp}.mp4")
                self.logger.info(f"Starting new video recording: {video_path}")
                process = (
                    ffmpeg
                    .input(self.camera_config['rtsp_url'], **self.args)
                    .filter('fps', fps=12)
                    .output(video_path, t=60, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast')
                    .overwrite_output()
                    .run()
                )
                self.logger.info(f"Video saved: {video_path}")
                self.video_queue.put(video_path)  # Add the video to the processing queue
            except Exception as e:
                self.logger.error("Error saving video stream")
                self.logger.error(e)

    def process_videos(self):
        """Thread for processing saved videos using the object counter."""
        while True:
            try:
                video_path = self.video_queue.get()
                if not video_path:
                    time.sleep(1)
                    continue
                
                self.logger.info(f"Processing video: {video_path}")
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                starttime = datetime.now().isoformat()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _ = self.counter.count(frame)
                    frame_count += 1

                self.counter.classwise_counts["starttime"] = starttime
                self.counter.classwise_counts["endtime"] = datetime.now().isoformat()
                self.counter.classwise_counts["cameraid"] = self.camera_config["camera_id"]
                
                success = self.send_post_request(self.counter.classwise_counts)
                if success:
                    self.logger.info(f"Successfully processed and sent data for video: {video_path}")
                    self.logger.info(f"Classwise Counts: {self.counter.classwise_counts}")
                else:
                    self.logger.warning(f"Failed to send data for video: {video_path}")
                    self.logger.info(f"Classwise Counts: {self.counter.classwise_counts}")

                self.counter.reset_count()
                cap.release()
                self.logger.info(f"Finished processing video: {video_path}")
            except Exception as e:
                self.logger.error("Error processing video")
                self.logger.error(e)

    def send_post_request(self, json_body, headers=None):
        """Sends data to the configured server URL."""
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(self.camera_config["url"], json=json_body, headers=headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error sending POST request: {e}")
            return False

    def run_threads(self):
        """Start the video-saving and object-counting threads."""
        self.logger.info("Starting threads for video saving and processing")
        video_thread = threading.Thread(target=self.save_video_stream, daemon=True)
        process_thread = threading.Thread(target=self.process_videos, daemon=True)
        video_thread.start()
        process_thread.start()
        video_thread.join()
        process_thread.join()

if __name__ == "__main__":
    rtspob = CAMERAMODEL()
    rtspob.run_threads()



# import os
# import cv2
# import time
# import ffmpeg
# import logging
# import requests
# import numpy as np
# from datetime import datetime
# from ObjectCount import ObjectCounter

# class CAMERAMODEL():

#     def __init__(self):

#         self.camera_config = {
#             "camera": "PaintCam1",
#             "camera_id": int("2"),
#             "region_points" : [(1250,0), (1250,1440)],
#             "url":"http://localhost:8000/ai/getcampayload",
#             "logdir":"D:\RohitDa\Camduc\paintai\logs",
#             "imgdir":"D:\RohitDa\Camduc\paintai\imgs",
#             "rtsp_url" : "rtsp://localhost:18554/mystream"
#         }

#         self.counter = ObjectCounter(
#             show=False,  # Display the output
#             region=self.camera_config['region_points'],  # Pass region points
#             model= "D:\RohitDa\Camduc\paintai\streamprocess\cameras\SudisaItem12K.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
#             # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
#             show_in=False,  # Display in counts
#             show_out=False,  # Display out counts
#             verbose=False
#             # line_width=2,  # Adjust the line width for bounding boxes and text display
#         )

#         if not os.path.exists(self.camera_config['logdir']):
#             os.makedirs(self.camera_config['logdir'])
#         if not os.path.exists(self.camera_config['imgdir']):
#             os.makedirs(self.camera_config['imgdir'])

#         self.logger = logging.getLogger(self.camera_config['camera'])
#         self.logger.setLevel(logging.DEBUG)
#         time_rotation = logging.handlers.TimedRotatingFileHandler(filename=os.path.join(self.camera_config['logdir'],self.camera_config['camera']+'.log'), when='W5',backupCount=1)
#         logFormat = logging.Formatter(
#             '%(asctime)s - %(name)s - %(threadName)s - %(funcName)s - %(levelname)s - %(message)s'
#             )
#         time_rotation.setFormatter(logFormat)
#         time_rotation.setLevel(logging.DEBUG)
#         self.logger.addHandler(time_rotation)
#         #self.camera_config = camera_config
#         self.rtsp_url = self.camera_config['rtsp_url']
#         self.update_duration = 60
#         self.args = {
#         "rtsp_transport": "tcp",
#         "fflags": "nobuffer",
#         "flags": "low_delay"
#         }
#         self.cache = {}
        
#         try:
#             self.logger.info("Begin Probing RTSP Stream for camera :- {}".format(self.rtsp_url))
#             probe = ffmpeg.probe(self.rtsp_url,**self.args)
#             cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
#             self.logger.info("fps: {}".format(cap_info['r_frame_rate']))
#             self.width = cap_info['width']          
#             self.height = cap_info['height']
#             self.logger.info(self.width,self.height)
#             up, down = str(cap_info['r_frame_rate']).split('/')
#             self.fps = eval(up) / eval(down)
#             self.logger.info(f"fps: {self.fps} and height:-{self.height} and width:- {self.width}")
        
#         except Exception as e:
#             self.logger.error("Failed to Probe RTSP Stream")
#             self.logger.error(e)
#             raise e

#     def send_post_request(self, json_body, headers=None):
#         if headers is None:
#             headers = {'Content-Type': 'application/json'}
#         try:
#             response = requests.post(self.camera_config["url"], json=json_body, headers=headers)
#             response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
#             return True  # Parse and return JSON response
#         except requests.exceptions.RequestException as e:
#             print(f"Error sending POST request: {e}")
#             return False

#     def process(self):
#         return (
#             ffmpeg
#             .input(self.rtsp_url, **self.args)
#             .filter('fps', fps=5)  # Add FPS filter to reduce the framerate to 6 FPS
#             .output('pipe:', format='rawvideo', pix_fmt='bgr24')
#             .overwrite_output()
#             .run_async(pipe_stdout=True)
#         )

#     def enqueue_frame_buffer(self):
#         self.process1 = self.process()
#         frame_count = 0
#         img_count = 0
#         starttime = datetime.now().isoformat()
#         while True:
#             try:
#                 in_bytes = self.process1.stdout.read(self.width * self.height * 3)
#                 # print(datetime.now(), in_bytes[:100]) # Print image shape if in_bytes
#                 if not in_bytes:
#                     self.logger.info("Some Issue with reading from STDOUT")
#                     self.logger.warning("Frame condition - ", datetime.now(), in_bytes[:100]) # Print image shape if not in_bytes
#                     time.sleep(20)
#                     self.process1.terminate()
#                     time.sleep(10)
#                     self.process1 = self.process()
#                     continue
                
#                 Frame = (
#                     np
#                     .frombuffer(in_bytes, np.uint8)
#                     .reshape([self.height, self.width, 3])
#                 )
#                 if Frame is not None:
#                     # cv2.imwrite(f"D:/RohitDa/Camduc/paintai/imgs/{img_count}.jpg", Frame)
#                     _ = self.counter.count(Frame)
#                     frame_count += 1
#                     img_count += 1
#                     if frame_count % (self.fps * self.update_duration) == 0:
#                         try:
#                             self.counter.classwise_counts["starttime"] = starttime
#                             self.counter.classwise_counts["endtime"] = datetime.now().isoformat()
#                             self.counter.classwise_counts["cameraid"] = self.camera_config["camera_id"]
#                             success = self.send_post_request(self.counter.classwise_counts)
#                             if success:
#                                 self.logger.info("Successfully sent to server")
#                                 # self.logger.info(("Classiwise Counts sent - ", self.counter.classwise_counts))
#                             else:
#                                 self.logger.info("Failed sent to server")
#                                 self.logger.info(("Classiwise Counts sent - ", self.counter.classwise_counts))
#                         except Exception as e:
#                             self.logger.error("Error Sending to Server")
#                             self.logger.error(e)
#                         frame_count = 0
#                         starttime = datetime.now().isoformat()
#                         self.counter.reset_count()

#             except Exception as e:
#                 self.logger.error("Problem with Processing")
#                 self.logger.error(e)

#     def run_threads(self):
#         self.logger.info("running threads")
#         self.enqueue_frame_buffer()

# if __name__ == "__main__":

#     rtspob = CAMERAMODEL()
#     rtspob.run_threads()
    
#     self.logger.info("=========================================")
