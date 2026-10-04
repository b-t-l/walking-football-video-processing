import os
import cv2
import sqlite3
import torch
import tqdm
import time
import math
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
import torch.nn.functional as F  # For resizing tensors
import torchvision.transforms as T
from ultralytics import YOLO
import supervision as sv
from utils import TimeUtils
from utils import ImageUtils
import gc
from team_assigner import TeamAssigner


########################################## - CLEAR GPU MEMORY IN PREP FOR NEXT BATCH
def cleanup_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


########################################## - DETERMINE THE DEVICE WE ARE WORKING WITH TO DO DETECTIONS
def get_device_config(DETECTION_FPS):
    """
    Dynamically determine the device and configure runtime settings.
    Returns: dict: Configuration dictionary with device, batch size, and precision settings.
    """
    if torch.cuda.is_available():
        device = "cuda"
        chunk_size = DETECTION_FPS*5  # Larger batch size for powerful NVIDIA GPUs - a100 - how many seconds of video to send to the batch before reseting
        batch_size = DETECTION_FPS*15  # Larger batch size for powerful NVIDIA GPUs - a100 - how many frames to process in a loop
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        use_half_precision = False
    elif torch.backends.mps.is_available():
        device = "cpu"  # m1 has a problem running :  Output channels > 65536 not supported at the MPS device
        chunk_size = 1
        batch_size = 1  # Smaller batch size for Apple M1 GPU - how many seconds of video to send to the batch before reseting
        use_half_precision = False
    else:
        device = "cpu"
        chunk_size = 1  #  - how many seconds of video to send to the batch before reseting
        batch_size = 1  # Small batch size for CPU fallback
        use_half_precision = False

    print(f"Running on device: {device}")
    print(f"Running on the following settings: chunk_size:{chunk_size} / batch_size:{batch_size}")
    return {
        "device": device,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "use_half_precision": use_half_precision,
    }




########################################## - CONVERT ANY FRAME_ID TO MM:SS
def frame_to_timestamp(frame_index,ORIGINAL_VIDEO_FPS):
    """
    Convert a frame index to MM:SS timestamp.
        
    Args:
        frame_index (int): The current frame index.
        fps (int): Frames per second of the video.
            
    Returns:
        str: Time in MM:SS format.
    """
    total_seconds = frame_index / ORIGINAL_VIDEO_FPS  # Get time in seconds
    minutes = int(total_seconds // 60)  # Get minutes
    seconds = int(total_seconds % 60)  # Remaining seconds
    return f"{minutes:02}:{seconds:02}"



########################################## - STORE DATA IN A DATASET FOR GPU BATCH COMPUTE:

class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


########################################## - CLASS DEFINITION
class CreateDatabase:

    def create_pitch_homography(self,pitch_keypoints):
        """
        Create homography matrix from pitch corners to normalized space
        """
        # Source points (actual pitch corners in image)
        src_points = np.float32([
            [next(kp for kp in pitch_keypoints if kp['name'] == 'left_bottom')['x'],
            next(kp for kp in pitch_keypoints if kp['name'] == 'left_bottom')['y']],
            [next(kp for kp in pitch_keypoints if kp['name'] == 'left_top')['x'],
            next(kp for kp in pitch_keypoints if kp['name'] == 'left_top')['y']],
            [next(kp for kp in pitch_keypoints if kp['name'] == 'right_top')['x'],
            next(kp for kp in pitch_keypoints if kp['name'] == 'right_top')['y']],
            [next(kp for kp in pitch_keypoints if kp['name'] == 'right_bottom')['x'],
            next(kp for kp in pitch_keypoints if kp['name'] == 'right_bottom')['y']]
        ])

        # Destination points (normalized rectangle)
        dst_points = np.float32([
            [0, 1],  # bottom left
            [0, 0],  # top left
            [1, 0],  # top right
            [1, 1]   # bottom right
        ])

        # Calculate homography matrix
        H, _ = cv2.findHomography(src_points, dst_points)
        return H

    def is_point_on_pitch(self, point, H):
        """
        Check if a point is on the pitch using homography
        """
        # Convert point to homogeneous coordinates
        p = np.array([[point[0], point[1], 1]], dtype=np.float32).T
        
        # Transform point using homography
        transformed = H @ p
        
        # Normalize coordinates
        x = transformed[0][0] / transformed[2][0]
        y = transformed[1][0] / transformed[2][0]
        
        # Check if point is within normalized space (with small margin for error)
        margin = 0.1  # Adjust this value based on your needs
        return (-margin <= x <= 1 + margin) and (-margin <= y <= 1 + margin)


    def is_detection_on_pitch(self, xmin, ymin, xmax, ymax, H):
        """
        Check if a detection is on the pitch by checking its bottom center point
        """
        # Use bottom center point of bounding box
        bottom_center_x = (xmin + xmax) / 2
        bottom_center_y = ymax
        
        return self.is_point_on_pitch((bottom_center_x, bottom_center_y), H)
    

    ########################################## - Update the tracker storage
    def save_tracker_state(self):
        """Save the current state of the BYTE tracker."""
        state = {
            'tracks': self.BYTE_TRACKER.tracked_tracks,
            'lost_tracks': self.BYTE_TRACKER.lost_tracks,
        }

        with open('../output/tracker_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        #print("✅ Tracker state saved successfully.")


    ########################################## - INIT FUNCTION
    def __init__(self, DURATION, 
                 BATCH_DURATION, 
                 DETECTION_FPS, 
                 EXCLUDE_TIME_RANGES,
                 DATABASE_FILE, 
                 OBJECT_IMAGES_FOLDER, 
                 TEAMS, 
                 GAME_RECORD, 
                 PITCH_DETECTION_OVERRIDE,
                 GAME_PITCH_KEYPOINTS_WITH_NAMES,
                 GAME_PITCH_BOUNDARIES):

        print("-------------------------------------------------")
        print("Create Database: Processing frame detections")

        self.INPUT_VIDEO_TITLE = GAME_RECORD['statistics_source_video']
        self.START_TIME = TimeUtils.time_to_seconds(GAME_RECORD['game_start_seconds'])
        self.DURATION = DURATION
        self.BATCH_DURATION = BATCH_DURATION
        self.DETECTION_FPS = DETECTION_FPS
        self.EXCLUDE_TIME_RANGES = EXCLUDE_TIME_RANGES

        self.DATABASE_FILE = DATABASE_FILE
        self.OBJECT_IMAGES_FOLDER = OBJECT_IMAGES_FOLDER
        self.TEAMS = TEAMS
        self.TEAM_ASSIGNER = TeamAssigner(TEAMS)


        self.PITCH_DETECTION_OVERRIDE = PITCH_DETECTION_OVERRIDE
        self.GAME_RECORD = GAME_RECORD
        self.GAME_PITCH_KEYPOINTS_WITH_NAMES = GAME_PITCH_KEYPOINTS_WITH_NAMES
        self.GAME_PITCH_BOUNDARIES = GAME_PITCH_BOUNDARIES
        self.PITCH_HOMOGRAPHY = self.create_pitch_homography(self.GAME_PITCH_KEYPOINTS_WITH_NAMES)

        # Dynamic device configuration
        config = get_device_config(self.DETECTION_FPS)
        self.TORCH_DEVICE = config["device"]
        self.BATCH_SIZE = config["batch_size"]
        self.CHUNK_SIZE = config["chunk_size"]
        self.USE_HALF_PRECISION = config["use_half_precision"]

        # Initialize YOLO models
        self.PLAYERS_MODEL_PATH = os.path.join('models', 'objects', 'best-v10-1920x1920.pt')
        self.BALL_MODEL_PATH = os.path.join('models', 'ball', 'best-v2-1920x1920.pt')
        self.TEAM_MODEL_PATH = os.path.join('models', 'teams', 'best-v3-192x192.pt')
        
        self.BALL_TRACKER = YOLO(self.BALL_MODEL_PATH)
        self.PLAYERS_TRACKER = YOLO(self.PLAYERS_MODEL_PATH)
        self.TEAM_TRACKER = YOLO(self.TEAM_MODEL_PATH)

        self.BYTE_TRACKER = sv.ByteTrack()
        self.save_tracker_state()

        self.KEYPOINT_NAMES = [
                "left_bottom",
                "left_top",
                "centre_top",
                "right_top",
                "right_bottom",
                "centre_bottom"
                ]
        self.KEYPOINTS_SKELETON = [
                [1,5],
                [5,4],
                [4,2],
                [2,1]
                ]

        # Retrieve imgsz from the models
        player_model_imgsz = self.PLAYERS_TRACKER.overrides.get('imgsz', (640, 640))
        if isinstance(player_model_imgsz, int):              # Ensure it's a tuple (H, W)
            player_model_imgsz = (player_model_imgsz, player_model_imgsz)  # Convert to (H, W) if single int
        height, width = player_model_imgsz
        self.PLAYER_MODEL_INPUT_SIZE = {'width':width,'height':height}
        print(f"-- YOLO PLAYERS_TRACKER INPUT SIZE: height={self.PLAYER_MODEL_INPUT_SIZE['height']}, width={self.PLAYER_MODEL_INPUT_SIZE['width']}")

        ball_model_imgsz = self.BALL_TRACKER.overrides.get('imgsz', (640, 640))
        if isinstance(ball_model_imgsz, int):              # Ensure it's a tuple (H, W)
            ball_model_imgsz = (ball_model_imgsz, ball_model_imgsz)  # Convert to (H, W) if single int
        height, width = ball_model_imgsz
        self.BALL_MODEL_INPUT_SIZE = {'width':width,'height':height}
        print(f"-- YOLO BALL_TRACKER INPUT SIZE: height={self.BALL_MODEL_INPUT_SIZE['height']}, width={self.BALL_MODEL_INPUT_SIZE['width']}")

        team_model_imgsz = self.TEAM_TRACKER.overrides.get('imgsz', (640, 640))
        if isinstance(team_model_imgsz, int):              # Ensure it's a tuple (H, W)
            team_model_imgsz = (team_model_imgsz, team_model_imgsz)  # Convert to (H, W) if single int
        height, width = team_model_imgsz
        self.TEAM_MODEL_INPUT_SIZE = {'width':width,'height':height}
        print(f"-- YOLO TEAM_TRACKER INPUT SIZE: height={self.TEAM_MODEL_INPUT_SIZE['height']}, width={self.TEAM_MODEL_INPUT_SIZE['width']}")


        # Open the video to get the orignal fps
        cap = cv2.VideoCapture(self.INPUT_VIDEO_TITLE)
        self.ORIGINAL_VIDEO_FPS = int(cap.get(cv2.CAP_PROP_FPS))  # Original video FPS
        self.ORIGINAL_VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ORIGINAL_VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare database and image folders
        self._prepare_environment()
        self.initialize_database()


    ########################################## - CREATE OS DIRECTORIES:
        
    def _prepare_environment(self):
        """Prepare the database and object images folder."""
        if os.path.exists(self.DATABASE_FILE):
            try:
                os.remove(self.DATABASE_FILE)
                print(f"Deleted existing database file: {self.DATABASE_FILE}")
            except Exception as e:
                print(f"Failed to delete {self.DATABASE_FILE}. Reason: {e}")
                return

        os.makedirs(self.OBJECT_IMAGES_FOLDER, exist_ok=True)
        for filename in os.listdir(self.OBJECT_IMAGES_FOLDER):
            file_path = os.path.join(self.OBJECT_IMAGES_FOLDER, filename)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


    ########################################## - INIT DATABASE: Create database and assign all the columns we will need:
                
    def initialize_database(self):

        """Initialize the SQLite database with the required schema."""
        conn = sqlite3.connect(self.DATABASE_FILE)
        cursor = conn.cursor()

        # game details:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS game (
            _id INTEGER PRIMARY KEY,
            game_id INTEGER,
            video STRING,
            video_fps INTEGER,
            detection_fps INTEGER 
        )
        ''')
        cursor.execute('''
            INSERT INTO game (_id, game_id, video, video_fps, detection_fps) VALUES (?, ?, ?, ?, ?)
            ''', (0, self.GAME_RECORD['game_id'],self.INPUT_VIDEO_TITLE, self.ORIGINAL_VIDEO_FPS, self.DETECTION_FPS))
        
        # player detections:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_objects (
            _id INTEGER PRIMARY KEY,
            frame_id INTEGER,
            xmin REAL,
            ymin REAL,
            xmax REAL,
            ymax REAL,
            confidence REAL,
            class_id REAL,
            tracker_id REAL,
            class_name TEXT,
            x_transformed_metres REAL,
            y_transformed_metres REAL,
            speed_km_per_hour REAL,
            total_distance_metres REAL,
            ball_posession BOOL,
            team INT,
            team_color TEXT,
            image_path TEXT
        )
        ''')

        # pitch detections:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_pitch (
            _id INTEGER PRIMARY KEY,
            frame_id INTEGER,
            x REAL,
            y REAL,
            name TEXT,
            confidence REAL,
            x_transformed_metres REAL,
            y_transformed_metres REAL
        )
        ''')

        conn.commit()
        conn.close()


    ########################################## - PITCH DETECTIONS:
    def process_pitch_detections(self, frames_batch, frame_ids):
        '''
        We are using a static pitch detection taken from the game records details: so we just pass back those to be saved and dont need to do any actual detection
        '''
        db_entries = []
        for idx, frame_id in enumerate(frame_ids):
            
            # Loop through each keypoint in GAME_PITCH_KEYPOINTS_WITH_NAMES
            for keypoint in self.GAME_PITCH_KEYPOINTS_WITH_NAMES:
                x = keypoint['x']
                y = keypoint['y']
                name = keypoint['name']
                conf_value = 1.0

                # Prepare the final data to go into the database
                db_entries.append({
                    'frame_id': frame_id,
                    'x': x,
                    'y': y,
                    'name': name,
                    'confidence': conf_value,
                    'x_transformed_metres': None,
                    'y_transformed_metres': None
                })

        # ✅ Write tracked detections to the database in bulk
        if db_entries:
            try:
                with sqlite3.connect(self.DATABASE_FILE) as conn:
                    cursor = conn.cursor()
                    for det in db_entries:
                        cursor.execute('''
                            INSERT INTO detected_pitch (
                                frame_id, x, y, confidence, name, x_transformed_metres, y_transformed_metres
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            det['frame_id'], det['x'], det['y'], det['confidence'], det['name'], det['x_transformed_metres'], det['y_transformed_metres']
                        ))
                    conn.commit()
            except Exception as e:
                print(f"❌ Failed to insert batch into database: {e}")


    ########################################## - PLAYER DETECTIONS:
    def process_player_detections(self, frames_batch, frame_ids):
        
        """
        Perform batch object detection, batch supervision conversion, batch tracking,
        and save results to the database efficiently.
        """
    
        # ✅ Batch YOLO detections
        if self.TORCH_DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                player_detections_batch = self.PLAYERS_TRACKER.predict(
                    frames_batch,
                    conf=0.3,
                    device=self.TORCH_DEVICE,
                    half=self.USE_HALF_PRECISION,
                    imgsz=(self.PLAYER_MODEL_INPUT_SIZE['height'],self.PLAYER_MODEL_INPUT_SIZE['width']),
                    verbose=False
                )
        else:
            player_detections_batch = self.PLAYERS_TRACKER.predict(
                frames_batch,
                conf=0.3,
                device=self.TORCH_DEVICE,
                half=self.USE_HALF_PRECISION,
                imgsz=(self.PLAYER_MODEL_INPUT_SIZE['height'],self.PLAYER_MODEL_INPUT_SIZE['width']),
                verbose=False
            )

        db_entries = []
        # loop each detection and prepare for db writing of detections:
        if len(player_detections_batch) > 0:

            for frame_idx, detection_result in enumerate(player_detections_batch):
                
                # convert to supervision
                frame_detections_supervision = sv.Detections.from_ultralytics(detection_result)
                # Add Tracking of Objects
                frame_detections_with_tracking = self.BYTE_TRACKER.update_with_detections(frame_detections_supervision)   
                # Loop results and get info from them that we want to store in db: Iterate over each detection
                for i in range(len(frame_detections_with_tracking.xyxy)):
                    
                    xmin, ymin, xmax, ymax = frame_detections_with_tracking.xyxy[i].tolist()
                    confidence = float(frame_detections_with_tracking.confidence[i])
                    class_id = int(frame_detections_with_tracking.class_id[i])
                    tracker_id = int(frame_detections_with_tracking.tracker_id[i]) if frame_detections_with_tracking.tracker_id is not None else None
                    class_name = frame_detections_with_tracking.data['class_name'][i] if 'class_name' in frame_detections_with_tracking.data else 'none'

                    # CHECK IF THE OBJECT IS ON THE PITCH (WE IGNORE ANY OBJECT OUTSIDE THE PITCH):
                    #if not self.is_detection_on_pitch(xmin, ymin, xmax, ymax, self.PITCH_HOMOGRAPHY):
                    #    continue  # Skip this detection as it's not on the pitch
                    
                    # DETERMINE THE TEAM CLASSIFIER:
                    team_name = None
                    team_color = None
                    # Apply the crop to image the detection is taken from using the calculated bounding box:
                    if int(ymin) < int(ymax) and int(xmin) < int(xmax):
                        
                        # Select the specific frame from batch and get the bbounding box as image (tensor)
                        frame_image = frames_batch[frame_idx]  # Shape: (channels, height, width)
                        bbox_image = frame_image[:, int(ymin):int(ymax), int(xmin):int(xmax)]
                            
                        if bbox_image.shape[1] > 0 and bbox_image.shape[2] > 0:  # Validate dimensions
                            
                            # resize it to the correct size for the team assigner model:
                            bbox_image_resized = ImageUtils.create_team_assigner_tensor(bbox_image,
                                                                                        self.TEAM_MODEL_INPUT_SIZE['width'],
                                                                                        self.TEAM_MODEL_INPUT_SIZE['height']
                                                                                        )

                            # do a team assignment via yolo object classification model:
                            team_detection = self.TEAM_TRACKER.predict(
                                bbox_image_resized,
                                conf=0.3,
                                device=self.TORCH_DEVICE,
                                half=self.USE_HALF_PRECISION,
                                imgsz=(192,192),
                                verbose=False
                            )

                            # store the resulting team name and color:
                            team_name, team_color = self.TEAM_ASSIGNER.assign_team(team_detection)
                            #print(f"{tracker_id}-{class_name}-{frame_idx}.jpg -  {frame_idx} -> {team_name}")
                            #print(f"Raw detection results: {team_detection}")

                            '''
                            # save me a few random frames of bounding box images so we can use them to improve the team assigner algo:
                            if frame_idx < self.DETECTION_FPS*2:
                                # Convert bbox_image to a NumPy array
                                bbox_image_numpy = bbox_image.cpu().numpy()
                                # Rescale values to [0, 255] if the pixel values are in [0, 1]
                                bbox_image_numpy = (bbox_image_numpy * 255).clip(0, 255).astype(np.uint8)
                                # Now apply transpose
                                bbox_image_numpy = bbox_image_numpy.transpose(1, 2, 0)
                                # save the images to the folder:
                                filename = f"{tracker_id}-{class_name}-{frame_idx}.jpg"
                                filepath = os.path.join(self.OBJECT_IMAGES_FOLDER, filename)
                                cv2.imwrite(filepath, bbox_image_numpy)
                            '''

                        else:
                            print(f"⚠️ Skipped invalid bounding box crop: {xmin},{ymin},{xmax},{ymax}")
                            team_name, team_color = None, None
                    else:
                        print(f"⚠️ Invalid bounding box coordinates detected: {xmin},{ymin},{xmax},{ymax}")
                        team_name, team_color = None, None


                    # scale the bounding boxes back to the original video size:
                    scaler_x = self.ORIGINAL_VIDEO_WIDTH/self.PLAYER_MODEL_INPUT_SIZE['width']
                    scaler_y = self.ORIGINAL_VIDEO_HEIGHT/self.PLAYER_MODEL_INPUT_SIZE['height']

                    db_entries.append({
                        'frame_id': frame_ids[frame_idx],
                        'xmin': xmin*scaler_x,
                        'ymin': ymin*scaler_y,
                        'xmax': xmax*scaler_x,
                        'ymax': ymax*scaler_y,
                        'confidence': confidence,
                        'class_id': class_id,
                        'tracker_id': tracker_id,
                        'class_name': class_name,
                        'x_transformed_metres': None,
                        'y_transformed_metres': None,
                        'speed_km_per_hour': None,
                        'total_distance_metres': None,
                        'team': str(team_name),
                        'team_color': str(team_color),
                        'ball_posession': False,
                        'image_path': ''
                    })
                    #print(class_name)
        else:
            print("⚠️ No player/referee/goalkeeper detections found in this batch.")
            return

        # ✅ Write tracked detections to the database in bulk
        with sqlite3.connect(self.DATABASE_FILE) as conn:
            cursor = conn.cursor()
            for det in db_entries:
                cursor.execute('''
                    INSERT INTO detected_objects (
                        frame_id, xmin, ymin, xmax, ymax, confidence, class_id, tracker_id, class_name,
                        x_transformed_metres, y_transformed_metres, speed_km_per_hour, total_distance_metres,
                        team, team_color, image_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    det['frame_id'], det['xmin'], det['ymin'], det['xmax'], det['ymax'], det['confidence'], det['class_id'], det['tracker_id'],
                    det['class_name'], det['x_transformed_metres'], det['y_transformed_metres'], det['speed_km_per_hour'],
                    det['total_distance_metres'], det['team'], det['team_color'], det['image_path']
                ))
            conn.commit()


    ########################################## - BALL DETECTIONS:
            
    def process_ball_detections(self, frames_batch, frame_ids):
        
        """
            Perform batch ball detection, filter only 'ball' detections,
            and save results to the database efficiently while preserving aspect ratio.
        """

        # ✅ Batch YOLO detections
        if self.TORCH_DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                ball_detections_batch = self.BALL_TRACKER.predict(
                    frames_batch,
                    conf=0.3,
                    device=self.TORCH_DEVICE,
                    half=self.USE_HALF_PRECISION,
                    imgsz=(self.BALL_MODEL_INPUT_SIZE['height'],self.BALL_MODEL_INPUT_SIZE['width']),
                    verbose=False
                )
        else:
            ball_detections_batch = self.BALL_TRACKER.predict(
                frames_batch,
                conf=0.3,
                device=self.TORCH_DEVICE,
                half=self.USE_HALF_PRECISION,
                imgsz=(self.BALL_MODEL_INPUT_SIZE['height'],self.BALL_MODEL_INPUT_SIZE['width']),
                verbose=False
            )
        
        db_entries = []
        # loop each detection and prepare for db writing of detections:
        if len(ball_detections_batch) > 0:

            for frame_idx, detection_result in enumerate(ball_detections_batch):
                
                # convert to supervision 
                frame_detections_supervision = sv.Detections.from_ultralytics(detection_result)
                #print(frame_detections_supervision)

                # Loop results and get info from them that we want to store in db: Iterate over each detection
                for i in range(len(frame_detections_supervision.xyxy)):
                    
                    xmin, ymin, xmax, ymax = frame_detections_supervision.xyxy[i].tolist()
                    confidence = float(frame_detections_supervision.confidence[i])
                    class_id = int(frame_detections_supervision.class_id[i])
                    #tracker_id = int(frame_detections_with_tracking.tracker_id[i]) if frame_detections_with_tracking.tracker_id is not None else None
                    class_name = frame_detections_supervision.data['class_name'][i] if 'class_name' in frame_detections_supervision.data else 'none'
                    #print(class_name)
                    
                    # CHECK IF THE OBJECT IS ON THE PITCH (WE IGNORE ANY OBJECT OUTSIDE THE PITCH):
                    #if not self.is_detection_on_pitch(xmin, ymin, xmax, ymax, self.PITCH_HOMOGRAPHY):
                    #    continue  # Skip this detection as it's not on the pitch

                    # scale the bounding boxes back to the original video size:
                    scaler_x = self.ORIGINAL_VIDEO_WIDTH/self.PLAYER_MODEL_INPUT_SIZE['width']
                    scaler_y = self.ORIGINAL_VIDEO_HEIGHT/self.PLAYER_MODEL_INPUT_SIZE['height']

                    db_entries.append({
                        'frame_id': frame_ids[frame_idx],
                        'xmin': xmin*scaler_x,
                        'ymin': ymin*scaler_y,
                        'xmax': xmax*scaler_x,
                        'ymax': ymax*scaler_y,
                        'confidence': confidence,
                        'class_id': class_id,
                        'tracker_id': None,
                        'class_name': class_name,
                        'x_transformed_metres': None,
                        'y_transformed_metres': None,
                        'speed_km_per_hour': None,
                        'total_distance_metres': None,
                        'team': None,
                        'team_color': None,
                        'ball_posession': False,
                        'image_path': ''
                    })
        
        else:
            #print("⚠️ No 'ball' detections found in this batch.")
            return

        # ✅ Write tracked detections to the database in bulk
        with sqlite3.connect(self.DATABASE_FILE) as conn:
            cursor = conn.cursor()
            for det in db_entries:
                if class_name == 'ball':
                    cursor.execute('''
                        INSERT INTO detected_objects (
                                frame_id, xmin, ymin, xmax, ymax, confidence, class_id, tracker_id, class_name,
                                x_transformed_metres, y_transformed_metres, speed_km_per_hour, total_distance_metres,
                                team, team_color, image_path
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        det['frame_id'], det['xmin'], det['ymin'], det['xmax'], det['ymax'], det['confidence'], det['class_id'], det['tracker_id'],
                        det['class_name'], det['x_transformed_metres'], det['y_transformed_metres'], det['speed_km_per_hour'],
                        det['total_distance_metres'], det['team'], det['team_color'], det['image_path']
                    ))
            conn.commit()

        #print("✅ Batch 'ball' detections saved to the database.")
 

    ############################ - TIME AND FRAME EXCLUSION FROM PROCESSING:
                
    def time_to_seconds(self,time_str):
        """Convert 'MM:SS' format to seconds."""
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds


    def get_exclusion_ranges_from_game_record(self, game_record):
        """
        Extract exclusion ranges from the GAME_RECORD variable.
        
        Args:
            game_record (dict): The game record dictionary containing time exclusions.
        
        Returns:
            list: List of (start, end) exclusion ranges in seconds.
        """
        exclusion_ranges = []

        # Predefined ranges from GAME_RECORD
        try:
            exclusion_ranges.append((0, self.time_to_seconds(game_record['game_start_seconds'])))
            exclusion_ranges.append((self.time_to_seconds(game_record['half_time_start_seconds']),
                                    self.time_to_seconds(game_record['half_time_end_seconds'])))
            exclusion_ranges.append((self.time_to_seconds(game_record['game_end_seconds']), float('inf')))
        except KeyError as e:
            print(f"❌ Missing key in GAME_RECORD: {e}")
        except ValueError as e:
            print(f"❌ Invalid time format in GAME_RECORD: {e}")

        # Add extra exclusions if provided
        if 'extra_exclusions' in game_record and pd.notna(game_record['extra_exclusions']):
            extra_exclusions = game_record['extra_exclusions'].split(',')
            for exclusion in extra_exclusions:
                try:
                    if exclusion.strip():  # Ensure the string isn't empty
                        start, end = exclusion.split('-')
                        exclusion_ranges.append((self.time_to_seconds(start.strip()), self.time_to_seconds(end.strip())))
                except ValueError:
                    print(f"⚠️ Invalid exclusion format: {exclusion}")
                except Exception as e:
                    print(f"⚠️ Error parsing exclusion '{exclusion}': {e}")

        print(f"📝 Final exclusion ranges: {exclusion_ranges}")
        return exclusion_ranges


    def should_skip_frame(self,current_time, exclusion_ranges):
        """Check if the current time falls in any exclusion range."""
        for start, end in exclusion_ranges:
            if start <= current_time <= end:
                return True
        return False
    
    def skip_frames_due_to_exclusion(self,cap, current_frame_id, frame_skip, exclusion_ranges, original_fps):
        """
        Skip frames based on exclusion ranges (e.g., halftime, breaks).
        """
        current_time = current_frame_id / original_fps
        if self.should_skip_frame(current_time, exclusion_ranges):
            print(f"⏭️ Skipping frame at time {frame_to_timestamp(current_frame_id, self.ORIGINAL_VIDEO_FPS)}")
            for _ in range(frame_skip - 1):  # Skip 'frame_skip - 1' frames
                ret, frame = cap.read()
                current_frame_id += 1
                if not ret:
                    print(f"⚠️ Skipped frame could not be read at {current_frame_id}, stopping skip.")
                    break
            current_frame_id += 1  # Increment for the skipped frame
            return current_frame_id, True  # Indicate frame was skipped
        
        return current_frame_id, False  # No skip happened


    def skip_frames_due_to_fps(self,cap, current_frame_id, frame_skip):
        """
        Skip frames to match the desired detection FPS.
        """
        #print(f"🎯 Skipping {frame_skip - 1} frames to match DETECTION_FPS")
        for _ in range(frame_skip - 1):
            ret, frame = cap.read()
            current_frame_id += 1
            if not ret:
                print(f"⚠️ Could not skip frame {current_frame_id}, stopping skip.")
                break
        current_frame_id += 1  # Move to the next valid frame
        return current_frame_id


    #################################################################################### RUN THE PROGRAM:
    def run(self):

        start_time = time.time()  # Start the timer for overall process time

        # Setup the video to use:
        cap = cv2.VideoCapture(self.INPUT_VIDEO_TITLE)
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📊 Video Info: FPS={original_fps}, Total Frames={total_frames}, Resolution={frame_width}x{frame_height}")
        print(f"🖥️ Processing on Device: {self.TORCH_DEVICE}, Batch Size: {self.BATCH_SIZE}, Precision: {'FP16' if self.USE_HALF_PRECISION else 'FP32'}")
        
        # Chunk the workload
        chunk_duration = self.CHUNK_SIZE  # seconds per chunk
        chunk_frames = chunk_duration * original_fps  # Number of frames per chunk

        # Determine from detection fps requested what our skip frames should be
        frame_skip = max(1, int(original_fps / self.DETECTION_FPS))
        print(f"🛠️ Frame Skip: {frame_skip}, ORIGINAL_FPS: {original_fps}, DETECTION_FPS: {self.DETECTION_FPS}")

        # Determine end frame based on DURATION
        if self.DURATION > 0:
            end_frame = min(int(self.DURATION * original_fps), total_frames)
        else:
            end_frame = total_frames
        print(f"🔄 Processing up to frame {end_frame} based on DURATION={self.DURATION} seconds")

        # Calculate total number of chunks to get thro all of the duration
        total_chunks = (end_frame + chunk_frames - 1) // chunk_frames  # Ceiling division to get total chunks

        # Load exclusion ranges from GAME_RECORD ( time we will be skipping from processing eg half time )
        exclusion_ranges = self.get_exclusion_ranges_from_game_record(self.GAME_RECORD)

        # GET VIDEO FRAMES: Get the frames for the chunk to process and prepare them (resize/tensor create):
        for chunk_idx, chunk_start_frame in enumerate(range(0, end_frame, chunk_frames), start=1):
            
            chunk_end_frame = min(chunk_start_frame + chunk_frames, end_frame)
            print(f"\n🗂️ Processing video chunk (video frames to tensors) {chunk_idx} of {total_chunks}: Frames {chunk_start_frame} to {chunk_end_frame}")

            frames = []
            frame_ids = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start_frame)
            current_frame_id = chunk_start_frame

            # loop thro each frome in the video in the chunk and skip where necessary due to detection fps or exclusion ranges:
            while current_frame_id < chunk_end_frame:
                
                #print(f"current_frame_id: {current_frame_id}")
                ret, frame = cap.read()
                
                if not ret:
                    print(f"⚠️ Could not read frame {current_frame_id}, skipping...")
                    #current_frame_id += frame_skip
                    break

                # Skip frames due to exclusion ranges
                current_frame_id, skipped = self.skip_frames_due_to_exclusion(cap, 
                                                                              current_frame_id, 
                                                                              frame_skip, 
                                                                              exclusion_ranges, 
                                                                              original_fps)
                if skipped:
                    continue
                
                # TRANSFORM FRAME INTO TENSOR: process the frame to the correct size to match detection size:
                frame_tensor = ImageUtils.create_tensor_and_resize(frame,self.PLAYER_MODEL_INPUT_SIZE['width'],
                                             self.PLAYER_MODEL_INPUT_SIZE['height'],
                                             self.ORIGINAL_VIDEO_WIDTH,
                                             self.ORIGINAL_VIDEO_HEIGHT
                                             )

                # ADD TO ARRAY/DICT we will process for this chunck
                frames.append(frame_tensor)
                frame_ids.append(current_frame_id)
                #print(f"✅ Adding Frame ID: {current_frame_id}")

                # LOOP: update to the new frame id and rerun: # Checking if we should skip the frame due to FPS
                current_frame_id = self.skip_frames_due_to_fps(cap, current_frame_id, frame_skip)

            if not frames:
                print("⚠️ No valid frames in this chunk, moving to the next chunk.")
                continue
            
            #for idx, frame in enumerate(frames):
            ##    filename = f"{idx}.jpg"
            #    filepath = os.path.join('check', filename)
            #    cv2.imwrite(filepath, frame)

            # DATASET CREATE: Move frames to device if CUDA/MPS is enabled
            dataset = FrameDataset(frames)
            dataloader = DataLoader(
                dataset,
                batch_size=self.BATCH_SIZE,
                num_workers=8,
                pin_memory=self.TORCH_DEVICE == "cuda"
            )

            # PROCESS DETECTIONS: loop thro each chunk processing 1 batch of frames at a time:
            print(f"🔄 Starting Detections for Chunk of video frames{chunk_idx} of {total_chunks}... (chunk size={chunk_duration} seconds)")
            with tqdm.tqdm(total=len(frames), desc=f"🔍 Processing Chunk {chunk_idx}/{total_chunks}", unit="frame") as pbar:
                try:

                    # BYTE TRACKER: Restore state before processing the next batch (if needed)
                    with open('../output/tracker_state.pkl', 'rb') as f:
                        state = pickle.load(f)
                        self.BYTE_TRACKER.tracked_tracks = state['tracks']
                        self.BYTE_TRACKER.lost_tracks = state['lost_tracks']

                    # LOOP: loop the batch and do the detections:
                    for batch_idx, batch_frames in enumerate(dataloader):
                        
                        # Move batch to the selected device
                        batch_frames = batch_frames.to(self.TORCH_DEVICE)  
                        if self.USE_HALF_PRECISION and self.TORCH_DEVICE == "cuda":
                            batch_frames = batch_frames.half()  # Use FP16 on CUDA if enabled

                        start_idx = batch_idx * self.BATCH_SIZE
                        end_idx = start_idx + len(batch_frames)

                        # PLAYHEAD POSITION FOR REFERENCE: in min:sec
                        current_frame_id = frame_ids[start_idx]
                        current_time = frame_to_timestamp(current_frame_id,self.ORIGINAL_VIDEO_FPS)
                        pbar.set_description(f"🔍 Processing Detections at video time: {current_time}")

                        # PROCESS: process the detections and save them to db:
                        self.process_player_detections(batch_frames, frame_ids[start_idx:end_idx])
                        self.process_ball_detections(batch_frames, frame_ids[start_idx:end_idx])
                        self.process_pitch_detections(batch_frames, frame_ids[start_idx:end_idx])

                        pbar.update(len(batch_frames))

                    # BYTE TRACKER: Save tracker state after processing a batch so we dont lose tracking when starting a new batch
                    self.save_tracker_state()

                finally:

                    # CLEAR MEMORY: Clear memory after each chunk
                    del frames, frame_ids, dataset, dataloader
                    torch.cuda.empty_cache() if self.TORCH_DEVICE == "cuda" else None
                    gc.collect()  # Force garbage collection

            # Cleanup after chunk
            cleanup_gpu_memory()

        cap.release()
        

        # TIME FOR FULL PROCESS: Calculate elapsed time for entire detections process:
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time
        minutes, seconds = divmod(int(total_time), 60)
        print(f"✅ Processing time: {minutes}:{seconds} sec")


