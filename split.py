from moviepy.video.io.VideoFileClip import VideoFileClip

# Load the video
video_path = "C:/Users/datacore/Downloads/recording_20250103_121617.mp4"  # Change to your actual video file
video = VideoFileClip(video_path)

# Get duration
total_duration = video.duration  # in seconds
segment_duration = total_duration / 6  # Divide into 6 equal parts

# Split the video into 6 parts
for i in range(6):
    start_time = i * segment_duration
    end_time = start_time + segment_duration
    output_path = f"output_part_{i+1}.mp4"
    
    # Extract subclip
    subclip = video.subclipped(start_time, end_time)
    subclip.write_videofile(output_path, codec="libx264", fps=video.fps)

# Close the video file
video.close()
