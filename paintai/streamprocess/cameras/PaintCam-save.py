import os
import time
import ffmpeg
import logging
import threading
from datetime import datetime, timedelta
from queue import Queue


class VideoSaver:
    def __init__(self):
        self.config = {
            "camera": "PaintCam-save",
            "logdir": "D:/Camduc-Sudisa/paintai/logs",
            "videodir": "D:/Camduc-Sudisa/paintai/videos",
            "rtsp_url": "rtsp://admin:hikvision%23123@192.168.3.66:554/Streaming/channels/101"
        }

        os.makedirs(self.config['logdir'], exist_ok=True)
        os.makedirs(self.config['videodir'], exist_ok=True)

        self.logger = logging.getLogger(self.config['camera'])
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(self.config['logdir'], 'saver.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.args = {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay"
        }

        self.video_queue = Queue()

    def save_video_stream(self):
        """Captures the RTSP stream and saves it as aligned hourly videos with a 59-second start grace period."""
        
        self.logger.info("Entered save_video_stream loop")
        print("Inside save_video_stream loop")

        while True:
            try:
                now = datetime.now()
                this_hour = now.replace(minute=0, second=0, microsecond=0)
                delay = (now - this_hour).total_seconds()

                if delay <= 59:
                    # Still in grace period for current hour, start recording
                    start_time = this_hour
                    self.logger.info(f"Within grace period ({delay:.2f}s). Recording for current hour starting at {start_time}")
                else:
                    # Too late for current hour, wait for next
                    start_time = this_hour + timedelta(hours=1)
                    wait_seconds = (start_time - now).total_seconds()
                    self.logger.info(f"Too late for current hour. Waiting {wait_seconds:.2f}s until next hour: {start_time}")
                    time.sleep(wait_seconds)

                    # Re-check after sleep
                    now = datetime.now()
                    delay = (now - start_time).total_seconds()
                    if delay > 59:
                        self.logger.warning(f"Start delay {delay:.2f}s exceeds grace period after wake-up. Skipping hour: {start_time}")
                        continue

                # Determine end time and actual recording duration
                end_time = start_time + timedelta(hours=1)
                remaining = (end_time - datetime.now()).total_seconds()

                if remaining <= 0:
                    self.logger.warning(f"No time left to record. Skipping hour: {start_time}")
                    continue

                # Define output path based on the start time of the intended hour
                timestamp = start_time.strftime('%Y%m%d_%H%M%S')
                video_path = os.path.join(self.config['videodir'], f"{timestamp}.mp4")

                self.logger.info(f"Starting video recording: {video_path} for {remaining:.2f} seconds")

                # Run ffmpeg to capture the stream
                ffmpeg.input(self.config['rtsp_url'], **self.args) \
                    .filter('fps', fps=10) \
                    .filter('scale', 1280, 720) \
                    .output(video_path, t=remaining, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast') \
                    .overwrite_output() \
                    .run()

                self.logger.info(f"Video saved successfully: {video_path}")
                self.video_queue.put(video_path)

            except Exception as e:
                self.logger.error(f"Error during video recording: {e}")

    def start(self):
        """Start video saving in a separate thread."""
        print("Starting the recording thread...")
        thread = threading.Thread(target=self.save_video_stream, daemon=True)
        thread.start()
        thread.join()

if __name__ == "__main__":
    print("PaintCam-save.py script started")
    video_saver = VideoSaver()
    video_saver.start()
