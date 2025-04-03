import os
import ffmpeg
import logging
import threading
from datetime import datetime
from queue import Queue

class VideoSaver:
    def __init__(self):
        self.config = {
            "camera": "PaintCamv6",
            "logdir": "D:/Camduc-Sudisa/paintai/logs",
            "videodir": "D:/Camduc-Sudisa/paintai/videos",
            "rtsp_url": "rtsp://admin:hikvision%23123@192.168.3.66:554/Streaming/channels/101"
        }

        os.makedirs(self.config['logdir'], exist_ok=True)
        os.makedirs(self.config['videodir'], exist_ok=True)

        self.logger = logging.getLogger(self.config['camera'])
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(self.config['logdir'], 'video_saver.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.args = {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay"
        }

        self.video_queue = Queue()

    def save_video_stream(self):
        """Captures the RTSP stream and saves it as hourly videos."""
        while True:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_path = os.path.join(self.config['videodir'], f"{timestamp}.mp4")
                self.logger.info(f"Starting new video recording: {video_path}")

                ffmpeg.input(self.config['rtsp_url'], **self.args) \
                    .filter('fps', fps=10) \
                    .filter('scale', 1280, 720) \
                    .output(video_path, t=3600, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast') \
                    .overwrite_output() \
                    .run()

                self.logger.info(f"Video saved: {video_path}")
                self.video_queue.put(video_path)
            except Exception as e:
                self.logger.error(f"Error saving video: {e}")

    def start(self):
        """Start video saving in a separate thread."""
        thread = threading.Thread(target=self.save_video_stream, daemon=True)
        thread.start()
        thread.join()

if __name__ == "__main__":
    video_saver = VideoSaver()
    video_saver.start()
