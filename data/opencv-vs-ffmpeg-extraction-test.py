import cv2
import ffmpeg
import numpy as np
import time
import subprocess

# Configuration
INPUT_VIDEO_TITLE = '../input_videos/polis-aw-womens.mp4'  # Replace with your video path
DURATION = 1  # Duration in seconds to extract frames
DETECTION_FPS = 6  # Desired detection frames per second
INPUT_VIDEO_FPS = 24  # Original video frames per second

# Calculate the frame interval based on the detection FPS
frame_interval = INPUT_VIDEO_FPS // DETECTION_FPS

def extract_frames_opencv():
    """Extract frames using OpenCV with profiling."""
    cap = cv2.VideoCapture(INPUT_VIDEO_TITLE)
    frames = []
    total_frames = int(DURATION * INPUT_VIDEO_FPS)

    start_time = time.time()
    for frame_id in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:  # Capture every nth frame
            frames.append(frame)
    cap.release()
    end_time = time.time()
    
    print(f"OpenCV extracted {len(frames)} frames in {end_time - start_time:.2f} seconds.")
    return frames

def extract_frames_with_ffmpeg(input_video, duration=1, fps=6):
    """Extract frames using FFmpeg and return them as NumPy arrays with profiling."""

    start_time = time.time()
    # Construct the FFmpeg command
    command = [
        'ffmpeg',
        '-hwaccel', 'videotoolbox',  # Use VideoToolbox for hardware acceleration
        '-i', input_video,
        '-t', str(duration),  # Duration in seconds
        '-vf', f'fps={fps}',  # Set the desired frames per second
        '-f', 'rawvideo',  # Output format
        '-pix_fmt', 'rgb24',  # Pixel format
        'pipe:1'  # Output to stdout
    ]

    # Run the command and capture the output
    try:
        process_start_time = time.time()  # Start profiling FFmpeg process
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()  # Capture the output and error
        process_end_time = time.time()  # End profiling FFmpeg process

        if process.returncode != 0:
            print(f"FFmpeg error: {err.decode()}")
            return None

        # Convert the output bytes to a NumPy array
        frame_size = (duration * fps, 2160, 3840, 3)  # Adjust based on your video resolution
        frames = np.frombuffer(out, np.uint8).reshape(frame_size)

        end_time = time.time()
        print(f"Extracted {len(frames)} frames at {fps} FPS for the first {duration} second(s).")
        print(f"FFmpeg extracted {len(frames)} frames in {end_time - start_time:.2f} seconds.")
        print(f"FFmpeg process time: {process_end_time - process_start_time:.2f} seconds.")
        return frames

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    print('checking version of opencv: ')
    print(cv2.__version__)

    # Check if the CUDA module is available
    if hasattr(cv2, 'cuda'):
        print("OpenCV has been built with CUDA support.")
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"Number of CUDA devices: {device_count}")
        for i in range(device_count):
            device_info = cv2.cuda.getDeviceProperties(i)
            print(f"Device {i}: {device_info.name}")
    else:
        print("OpenCV does not have CUDA support.")

    print("Starting frame extraction with OpenCV...")
    opencv_frames = extract_frames_opencv()

    print("Starting frame extraction with FFmpeg...")
    ffmpeg_frames = extract_frames_with_ffmpeg(INPUT_VIDEO_TITLE, DURATION, DETECTION_FPS)

if __name__ == "__main__":
    main()