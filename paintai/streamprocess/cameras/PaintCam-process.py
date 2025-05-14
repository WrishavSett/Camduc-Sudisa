import os
import cv2
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from ObjectCount import ObjectCounter

class VideoProcessor:
    def __init__(self):
        self.config = {
            "camera": "PaintCam-process",
            "camera_id": 2,
            "region_points": [(100, 200), (1180, 200), (1180, 500), (100, 500)],
            "url": "http://localhost:8000/ai/getcampayload",
            "logdir": "D:/Camduc-Sudisa/paintai/logs",
            "processed_logdir": "D:/Camduc-Sudisa/paintai/processed_logs",
            "videodir": "D:/Camduc-Sudisa/paintai/videos",
            "model_path": "D:/Camduc-Sudisa/Iter3.4_Sub.pt"
        }

        os.makedirs(self.config['logdir'], exist_ok=True)
        os.makedirs(self.config['processed_logdir'], exist_ok=True)

        self.logger = logging.getLogger(self.config['camera'])
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(self.config['logdir'], 'processor.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.counter = ObjectCounter(
            show=False,
            region=self.config['region_points'],
            model=self.config['model_path'],
            show_in=False,
            show_out=False,
            verbose=False,
            conf=0.5
        )

    def scan_and_process(self):
        """Continuously scans for new videos and processes them."""
        try:
            while True:
                video_files = [f for f in os.listdir(self.config["videodir"]) if f.endswith(".mp4")]

                for video_file in sorted(video_files):  # Sort for chronological processing
                    video_path = os.path.join(self.config["videodir"], video_file)
                    processed_file_path = os.path.join(self.config["processed_logdir"], f"{video_file}.json")

                    # Skip already processed videos
                    if os.path.exists(processed_file_path):
                        print(f"[SKIP] Already processed: {video_file}")
                        continue

                    # Extract start time from filename
                    try:
                        basename = os.path.splitext(video_file)[0]
                        start_time = datetime.strptime(basename, "%Y%m%d_%H%M%S")
                        print(f"[INFO] Checking video: {video_file}, Start Time Parsed: {start_time}")
                    except ValueError:
                        self.logger.warning(f"Filename format incorrect, skipping: {video_file}")
                        continue

                    # Skip videos for current or incomplete hour
                    now = datetime.now()
                    hour_end_time = start_time + timedelta(hours=1, seconds=60)  # Add 60s buffer

                    if now < hour_end_time:
                        self.logger.info(f"Skipping {video_file} — still within the hour + 60s buffer.")
                        print("[WAIT] " + f"Skipping {video_file} — still within the hour + 60s buffer.")
                        continue

                    # # Skip very recent files to ensure .mp4 is fully closed
                    # if (time.time() - os.path.getmtime(video_path)) < 10:
                    #     self.logger.info(f"Skipping {video_file} — file too recent.")
                    #     continue

                    # Process the video
                    print(f"[PROCESS] Starting processing for: {video_file}")
                    self.logger.info(f"Processing video: {video_path}")
                    counts = self.process_video(video_path)

                    if counts:
                        with open(processed_file_path, "w") as f:
                            json.dump(counts, f, indent=4)
                        print(f"[SUCCESS] Saved processed data: {processed_file_path}")
                        self.logger.info(f"Saved processed data to: {processed_file_path}")

                    time.sleep(2)

        except KeyboardInterrupt:
            print("\n[EXIT] Ctrl+C pressed. Exiting gracefully...")
            self.logger.info("KeyboardInterrupt received. Exiting processing loop.")

    def process_video(self, video_path):
        """Processes a single video file and counts objects."""
        try:
            self.counter.reset_count()
            cap = cv2.VideoCapture(video_path)
            starttime = datetime.now().isoformat()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _ = self.counter.count(frame)

            self.counter.classwise_counts["starttime"] = starttime
            self.counter.classwise_counts["endtime"] = datetime.now().isoformat()
            self.counter.classwise_counts["cameraid"] = self.config["camera_id"]

            success = self.send_post_request(self.counter.classwise_counts)
            if success:
                self.logger.info(f"Successfully sent data for video: {video_path}")
                self.logger.info(f"Classwise Counts: {self.counter.classwise_counts}")
            else:
                self.logger.warning(f"Failed to send data for video: {video_path}")
                self.logger.info(f"Classwise Counts: {self.counter.classwise_counts}")

            cap.release()
            return self.counter.classwise_counts

        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            return None

    def send_post_request(self, json_body):
        """Sends object count data to the server."""
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(self.config["url"], json=json_body, headers=headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error sending POST request: {e}")
            return False

    def start(self):
        """Starts scanning for videos to process."""
        self.scan_and_process()

if __name__ == "__main__":
    print("PaintCam-process.py Process Started. Press Ctrl+C to Stop.")
    video_processor = VideoProcessor()
    video_processor.start()
