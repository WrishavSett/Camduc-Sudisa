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
        MAX_RETRIES = 5

        while True:
            try:
                now = datetime.now()
                this_hour = now.replace(minute=0, second=0, microsecond=0)
                delay = (now - this_hour).total_seconds()

                if delay <= 59:
                    start_time = this_hour
                    self.logger.info(f"Within grace period ({delay:.2f}s). Recording for current hour starting at {start_time}")
                else:
                    start_time = this_hour + timedelta(hours=1)
                    wait_seconds = (start_time - now).total_seconds()
                    self.logger.info(f"Too late for current hour. Waiting {wait_seconds:.2f}s until next hour: {start_time}")
                    time.sleep(wait_seconds)

                    now = datetime.now()
                    delay = (now - start_time).total_seconds()
                    if delay > 59:
                        self.logger.warning(f"Start delay {delay:.2f}s exceeds grace period after wake-up. Skipping hour: {start_time}")
                        continue

                end_time = start_time + timedelta(hours=1)
                remaining = (end_time - datetime.now()).total_seconds()

                if remaining <= 0:
                    self.logger.warning(f"No time left to record. Skipping hour: {start_time}")
                    continue

                timestamp = start_time.strftime('%Y%m%d_%H%M%S')
                video_path = os.path.join(self.config['videodir'], f"{timestamp}.mp4")
                self.logger.info(f"Starting video recording: {video_path} for {remaining:.2f} seconds")

                retry_count = 0
                while retry_count < MAX_RETRIES:
                    try:
                        ffmpeg.input(self.config['rtsp_url'], **self.args) \
                            .filter('fps', fps=10) \
                            .filter('scale', 1280, 720) \
                            .output(video_path, t=remaining, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast') \
                            .overwrite_output() \
                            .run()

                        self.logger.info(f"Video saved successfully: {video_path}")
                        self.video_queue.put(video_path)
                        break  # success
                    except ffmpeg.Error as fferr:
                        retry_count += 1
                        self.logger.error(f"[RETRY {retry_count}/{MAX_RETRIES}] ffmpeg error while saving {video_path}: {fferr}")
                        wait_time = 5 * (2 ** (retry_count - 1))  # exponential backoff
                        self.logger.info(f"Waiting {wait_time}s before retrying.")
                        time.sleep(wait_time)

                        if retry_count == MAX_RETRIES:
                            self.logger.error(f"Failed to record {video_path} after {MAX_RETRIES} retries. Skipping.")
            except Exception as e:
                self.logger.error(f"Unexpected error in video loop: {e}")

    # def start(self):
    #     """Start video saving in a separate thread."""
    #     thread = threading.Thread(target=self.save_video_stream, daemon=True)
    #     thread.start()
    #     thread.join()

    def start(self):
        """Start video saving in a separate thread."""
        thread = threading.Thread(target=self.save_video_stream, daemon=True)
        thread.start()

        # Keep the main thread alive manually to catch Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting Down")
            self.logger.info("KeyboardInterrupt received. Stopping PaintCam-save.py Process.")


if __name__ == "__main__":
    print("PaintCam-save.py Process Started. Press Ctrl+C to Stop.")
    video_saver = VideoSaver()
    video_saver.start()
