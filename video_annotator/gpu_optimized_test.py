import os
import cv2
print("CUDA support:", hasattr(cv2, 'cuda'))
import time
import sqlite3
import subprocess
import numpy as np
from utils import *
from video_annotator import draw_annotations
import torch

class AnnotatedVideoProcessor:
    def __init__(self, GAME_RECORD, input_video, output_video, DATABASE_FILE, detection_fps, use_gpu=True, chunk_size=300, start_frame=None, end_frame=None):
        """
        Initialize the processor with input and output details, database, and processing settings.
        
        Args:
            input_video (str): Path to the input video file.
            output_video (str): Path to save the final output video.
            database_path (str): Path to the SQLite database containing detection data.
            detection_fps (int): The desired FPS for processing and output video.
            use_gpu (bool): Whether to use GPU acceleration for processing.
            chunk_size (int): Number of frames to process in a single chunk.
            start_frame (int or None): First frame to process (inclusive).
            end_frame (int or None): Last frame to process (inclusive).
        """
        self.INPUT_VIDEO_TITLE = GAME_RECORD['statistics_source_video']
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.OUTPUT_VIDEO_TITLE = f"../output/game-id-{GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(GAME_RECORD['title'])}-{timestr}.mp4"

        self.DB_PATH = DATABASE_FILE
        self.DETECTION_FPS = 6

        # Determine the device and configure runtime settings:
        if torch.cuda.is_available():
            self.USE_GPU = True
            self.CHUNK_SIZE = self.DETECTION_FPS*4*4

        elif torch.backends.mps.is_available():
            self.USE_GPU = False
            self.CHUNK_SIZE = self.DETECTION_FPS*4
        else:
            self.USE_GPU = False
            self.CHUNK_SIZE = self.DETECTION_FPS*4
        
        print(f"Using cuda gpu: {self.USE_GPU}")

        self.START_FRAME = start_frame  # Start frame (inclusive)
        self.END_FRAME = end_frame      # End frame (inclusive)

        # Create temp directory for video chunks
        #self.TEMP_DIR = "./temp_video_chunks"
        #os.makedirs(self.TEMP_DIR, exist_ok=True)
        # Create and clean temporary directory for chunks
        self.TEMP_DIR = "../output/temp_video_chunks"
        self.clear_temporary_directory_files()
        


    def clear_temporary_directory_files(self):
        start_time = time.time()
        if os.path.exists(self.TEMP_DIR):
            # Remove all files in the directory
            for file in os.listdir(self.TEMP_DIR):
                file_path = os.path.join(self.TEMP_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"🔄 Deleted existing temp file: {file}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        # Create the directory (or ensure it exists if already empty)
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        print(f"Cleared temporary directory in {time.time() - start_time:.2f} seconds.")


    def get_frame_details_from_db(self, frame_ids):
        """Fetch detection data for a given list of frame IDs."""
        start_time = time.time()
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Query database for the frame IDs in the current chunk
        query = f"""
        SELECT frame_id, xmin, ymin, xmax, ymax, confidence, class_id, tracker_id, class_name, 
                x_transformed_metres, y_transformed_metres, speed_km_per_hour, 
                total_distance_metres, ball_posession, is_inside_pitch, team, team_color
        FROM detected_objects 
        WHERE frame_id IN ({','.join(map(str, frame_ids))}) AND is_inside_pitch = 1
        """
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        # Organize data by frame_id
        detections = {}
        for row in results:
            frame_id = row[0]
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append({
                'xmin': row[0],
                'ymin': row[1],
                'xmax': row[2],
                'ymax': row[3],
                'confidence': row[4],
                'class_id': row[5],
                'tracker_id': row[6],
                'class_name': row[7],
                'x_transformed_metres': row[8],
                'y_transformed_metres': row[9],
                'speed_km_per_hour': row[10],
                'total_distance_metres': row[11],
                'ball_posession': row[12],
                'is_inside_pitch': row[13],
                'team': row[14],
                'team_color': row[15]
            })
        print(f"Fetched frame details from DB in {time.time() - start_time:.2f} seconds.")
        return detections

    def annotate_frame(self, frame, detections):
        """Draw annotations on the frame using OpenCV."""
        start_time = time.time()
        # Convert frame to GPU memory if using GPU
        if self.USE_GPU:
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)

            for det in detections:
                color = tuple(map(int, det["team_color"].strip("()").split(","))) if det["team_color"] else (0, 255, 0)
                cv2.cuda.rectangle(frame_gpu, (int(det["xmin"]), int(det["ymin"])), (int(det["xmax"]), int(det["ymax"])), color, 2)
                label = f"{det['class_name']} ({det['team']})"
                cv2.cuda.putText(frame_gpu, label, (int(det["xmin"]), int(det["ymin"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Download the processed frame back to CPU
            frame = frame_gpu.download()
        else:
            for det in detections:
                color = tuple(map(int, det["team_color"].strip("()").split(","))) if det["team_color"] else (0, 255, 0)
                cv2.rectangle(frame, (int(det["xmin"]), int(det["ymin"])), (int(det["xmax"]), int(det["ymax"])), color, 2)
                label = f"{det['class_name']} ({det['team']})"
                cv2.putText(frame, label, (int(det["xmin"]), int(det["ymin"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #print(f"Annotated frame in {time.time() - start_time:.2f} seconds.")
        return frame

    def process_chunks(self):
        """Process video chunks based on frame IDs."""
        start_time = time.time()
        print(f"🔄---- Processing video: {self.INPUT_VIDEO_TITLE}")

        # Open video to get metadata
        cap = cv2.VideoCapture(self.INPUT_VIDEO_TITLE)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📊 Video FPS: {original_fps}, Total Frames: {total_frames}")
        print(f"🎯 Target FPS: {self.DETECTION_FPS}")
        print(f"🎯 Chunk size: {self.CHUNK_SIZE }")

        # Define the range of frames to process
        start_frame = self.START_FRAME if self.START_FRAME is not None else 0
        end_frame = self.END_FRAME if self.END_FRAME is not None else total_frames - 1
        print(f"🔄 Processing range: {start_frame} to {end_frame}")

        # Get frame IDs within the range
        frame_ids = range(start_frame, end_frame + 1, int(original_fps / self.DETECTION_FPS))

        # Split frame IDs into chunks
        chunk_files = []
        for chunk_idx, chunk_start in enumerate(range(0, len(frame_ids), self.CHUNK_SIZE)):
            chunk_end = min(chunk_start + self.CHUNK_SIZE, len(frame_ids))
            chunk_frame_ids = list(frame_ids[chunk_start:chunk_end])
            print(f"🔄 Processing Chunk {chunk_idx + 1}: Frames {chunk_start} to {chunk_end}")

            # Fetch detection data for the chunk
            detections = self.get_frame_details_from_db(chunk_frame_ids)

            # Process and annotate each frame
            frames = []
            total_annotation_time = 0  # Initialize total annotation time for the chunk
            for frame_id in chunk_frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                success, frame = cap.read()
                if not success:
                    break
                if frame_id in detections:
                    annotation_start_time = time.time()  # Start timing annotation
                    frame = self.annotate_frame(frame, detections[frame_id])
                    annotation_time = time.time() - annotation_start_time  # Calculate annotation time
                    total_annotation_time += annotation_time  # Accumulate annotation time
                frames.append(frame)

            # Save the chunk as a video file
            chunk_file = os.path.join(self.TEMP_DIR, f"chunk_{chunk_idx + 1:03d}.mp4")
            chunk_files.append(chunk_file)
            self.save_chunk(frames, chunk_file)

            # Print total annotation time for the chunk
            print(f"Total annotation time for Chunk {chunk_idx + 1}: {total_annotation_time:.2f} seconds")

        cap.release()

        # Merge all chunks into a final video
        print(f"🔄 ------- Merging video chunks:")
        self.merge_chunks(chunk_files)
        print(f"🔄 ------- COMPLETED:")
        print(f"✅ Completed in {time.time() - start_time:.2f} seconds")


    def save_chunk(self, frames, output_path):
        """Save a list of frames as a video chunk."""
        start_time = time.time()
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.DETECTION_FPS, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Saved chunk in {time.time() - start_time:.2f} seconds.")

    def merge_chunks(self, chunk_files):
        """Merge video chunks into a single file using FFmpeg."""
        start_time = time.time()
        concat_file = os.path.join('../output/', "concat_list.txt")
        with open(concat_file, "w") as f:
            for chunk in chunk_files:
                f.write(f"file '{chunk}'\n")

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c", "copy", self.OUTPUT_VIDEO_TITLE
        ]
        if self.USE_GPU:
            ffmpeg_cmd.insert(2, "-hwaccel")
            ffmpeg_cmd.insert(3, "cuda")

        subprocess.run(ffmpeg_cmd)
        print(f"Merged chunks in {time.time() - start_time:.2f} seconds.")
        print(f"✅ Final video saved: {self.OUTPUT_VIDEO_TITLE}")
