import os
import cv2
import sqlite3
import torch
import tqdm
import time
import logging
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
from matplotlib.path import Path
from shapely.geometry import Point, Polygon
from utils import TimeUtils
from utils import ImageUtils
import gc
import platform
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from team_assigner import TeamAssigner


########################################## - CLEAR GPU MEMORY IN PREP FOR NEXT BATCH
def cleanup_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


########################################## - DETERMINE THE DEVICE WE ARE WORKING WITH TO DO DETECTIONS
# On a Mac the two big YOLO models (players, ball) run through Core ML (GPU / Neural Engine) instead of the CPU.
# Benchmarked on game 22 (M1 Max, 1920x1920 stretched input, same as training): ~130 ms/frame per model instead of
# ~790 ms, with every detection matched (recall/precision 100%, box IoU 0.99 players / 0.88 ball).
# The Core ML package is exported once, next to the .pt, and re-exported if the .pt is newer. If anything fails
# (not a Mac, coremltools missing, class names differ) the normal PyTorch model is used, so nothing breaks.
# Set to False to force the PyTorch/CPU path.
USE_COREML_ON_MAC = False

# With a recent PyTorch (2.14+, checked on torch 2.14.1 / ultralytics 8.4.174) the Apple GPU (MPS) runs the 1920x1920
# models directly: ~87 ms/frame instead of ~548 ms on CPU, with detections identical to the CPU run (IoU 1.00).
# Older PyTorch (2.5) failed with "Output channels > 65536". A quick test run at start-up checks MPS really works
# and falls back to the CPU if not. Core ML (above) is only used if USE_COREML_ON_MAC is switched on.
USE_MPS_ON_MAC = True
MPS_BATCH_SIZE = 1      # frames per predict() call on MPS. Raise (try 2, 4) once the baseline run works, and compare timings.
MPS_HALF = True         # FP16 on the GPU: ~73 ms instead of 87 ms per model per frame. Benchmark (game 22): every box still found
                        # (259/259 players, 21/21 ball); box overlap with the FP32 result IoU 0.98 players / 0.84 ball (sub-pixel shifts).
MPS_TEAM_ON_GPU = True  # tiny 192x192 team classifier on the GPU too (False = CPU; test showed CPU 30 ms/frame)


def load_predictor(pt_model, pt_path, imgsz=1920):
    """Return the model object to call .predict() on: a Core ML version of the .pt on Apple hardware, else the .pt model."""
    if not (USE_COREML_ON_MAC and platform.system() == "Darwin" and not torch.cuda.is_available()):
        return pt_model, False
    package = os.path.splitext(pt_path)[0] + ".mlpackage"
    try:
        if (not os.path.exists(package)) or os.path.getmtime(package) < os.path.getmtime(pt_path):
            print(f"-- Exporting {pt_path} to Core ML at {imgsz}x{imgsz} FP16 (one-off, takes a minute) ...")
            package = str(YOLO(pt_path).export(format="coreml", imgsz=imgsz, half=True, nms=False))
        model = YOLO(package, task="detect")
        if dict(model.names) != dict(pt_model.names):
            print(f"⚠️ Core ML class names differ from {pt_path} - using PyTorch/CPU for this model")
            return pt_model, False
        print(f"-- {pt_path}: running through Core ML ({package})")
        return model, True
    except Exception as e:
        print(f"⚠️ Core ML not available for {pt_path} ({type(e).__name__}: {str(e)[:120]}) - using PyTorch/CPU")
        return pt_model, False


def mps_works(models_and_sizes, half=False):
    """Run one blank frame through each model on the Apple GPU. True only if every model runs without error."""
    try:
        for model, imgsz in models_and_sizes:
            blank = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)
            model.predict(blank, device="mps", imgsz=(imgsz, imgsz), verbose=False, **({"half": True} if half else {}))
        return True
    except Exception as e:
        print(f"⚠️ MPS (Apple GPU) test failed ({type(e).__name__}: {str(e)[:150]})")
        return False


def get_device_config(DETECTION_FPS):
    """
    Dynamically determine the device and configure runtime settings.
    Returns: dict: Configuration dictionary with device, batch size, and precision settings.
    """
    if torch.cuda.is_available():
        device = "cuda"
        chunk_size = 30  # Larger batch size for powerful NVIDIA GPUs - a100 - how many seconds of video to send to the batch before reseting
        batch_size = 90  # Larger batch size for powerful NVIDIA GPUs - a100 - how many frames to process in a loop
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        use_half_precision = False
    elif torch.backends.mps.is_available():
        device = "mps" if USE_MPS_ON_MAC else "cpu"  # old PyTorch (<2.14) failed on MPS: "Output channels > 65536"; start-up check below falls back to cpu
        chunk_size = 1
        batch_size = MPS_BATCH_SIZE if USE_MPS_ON_MAC else 1  # frames per predict() call
        use_half_precision = bool(MPS_HALF and USE_MPS_ON_MAC)
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


    
    ########################################## - Check if a bounding box is inside the shape of the field:
    def is_detection_box_inside_pitch(self, frame_id, class_name, x_min, y_min, x_max, y_max):
        # Define the polygon (pitch) from your points
        '''
        polygon_points = [
            (305, 1119),  # left_bottom
            (1017, 230),  # left_top
            (1927, 236),  # centre_top
            (2865, 214),  # right_top
            (3641, 1140),  # right_bottom
            (1927, 1110)   # centre_bottom
        ]
        '''
        polygon_points = self.GAME_PITCH_KEYPOINTS
        #print(f"Polygon Points: {polygon_points}")

        # Create a Shapely Polygon object
        if getattr(self, '_pitch_polygon', None) is None:
            self._pitch_polygon = Polygon(polygon_points)
        shapely_polygon = self._pitch_polygon

        # Determine the placement of the bounding box
        placement_x = ((x_max - x_min) / 2 + x_min)
        placement_y = ((y_max - y_min) / 2 + y_min)
        object_placement = 'centre'
        x1, y1 = 1927, 236
        x2, y2 = 1927, 1110
        cross_product = (x2 - x1) * (placement_y - y1) - (y2 - y1) * (placement_x - x1)
        if cross_product > 0:
            object_placement = 'left'
        elif cross_product < 0:
            object_placement = 'right'
        else:
            object_placement = 'centre'

        # Select the test point based on the placement
        if object_placement == 'left':
            test_point_coords = (x_max, y_max)
        elif object_placement == 'right':
            test_point_coords = (x_min, y_max)
        else:  # Center
            test_point_coords = (((x_min + x_max) / 2), y_max)

        # Convert to Shapely Point object
        test_point = Point(float(test_point_coords[0]), float(test_point_coords[1]))

        # Check if the test point is inside or intersects the polygon
        is_inside = shapely_polygon.intersects(test_point)  # Use intersects instead of contains

        # Debugging output
        #print(f"Frame id: {frame_id} | Class: {class_name}  | Test Point: {test_point_coords}, Placement: {object_placement} = Is Inside: {is_inside}")

        return is_inside
    


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
                 GAME_PITCH_KEYPOINTS,
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
        self.GAME_PITCH_KEYPOINTS = GAME_PITCH_KEYPOINTS
        self.GAME_PITCH_BOUNDARIES = GAME_PITCH_BOUNDARIES

        # Dynamic device configuration
        config = get_device_config(self.DETECTION_FPS)
        self.TORCH_DEVICE = config["device"]
        self.BATCH_SIZE = config["batch_size"]
        self.CHUNK_SIZE = config["chunk_size"]
        self.USE_HALF_PRECISION = config["use_half_precision"]
        self.TIMINGS = {}          # seconds spent per stage, printed at the end of run()
        self.TIMINGS_BG = {}       # stages that run in the background reader thread (overlapped)
        self.TIMED_FRAMES = 0
        _inside_pitch_check = self.is_detection_box_inside_pitch
        def _timed_inside_pitch_check(*a, **k):
            _t = time.perf_counter()
            r = _inside_pitch_check(*a, **k)
            self._tick('inside-pitch check', _t)
            return r
        self.is_detection_box_inside_pitch = _timed_inside_pitch_check

        # Initialize YOLO models
        self.PLAYERS_MODEL_PATH = os.path.join('models', 'objects', 'best-v10-1920x1920.pt')
        self.BALL_MODEL_PATH = os.path.join('models', 'ball', 'best-v2-1920x1920.pt')
        self.TEAM_MODEL_PATH = os.path.join('models', 'teams', 'best-v7-192x192.pt')
        
        self.BALL_TRACKER = YOLO(self.BALL_MODEL_PATH)
        self.PLAYERS_TRACKER = YOLO(self.PLAYERS_MODEL_PATH)
        self.TEAM_TRACKER = YOLO(self.TEAM_MODEL_PATH)

        # the models that actually run predict() for players / ball (Core ML on a Mac, else the same .pt models).
        # The .pt models above are still used to read the input sizes.
        self.PLAYERS_PREDICTOR, _players_coreml = load_predictor(self.PLAYERS_TRACKER, self.PLAYERS_MODEL_PATH)
        self.BALL_PREDICTOR, _ball_coreml = load_predictor(self.BALL_TRACKER, self.BALL_MODEL_PATH)
        if self.TORCH_DEVICE == "mps":
            if mps_works([(self.PLAYERS_PREDICTOR, 1920), (self.BALL_PREDICTOR, 1920)], half=self.USE_HALF_PRECISION):
                print(f"-- Detections running on the Apple GPU (MPS), batch size {self.BATCH_SIZE}")
            elif self.USE_HALF_PRECISION and mps_works([(self.PLAYERS_PREDICTOR, 1920), (self.BALL_PREDICTOR, 1920)]):
                print("-- FP16 failed on MPS: using FP32 on the Apple GPU instead")
                self.USE_HALF_PRECISION = False
            else:
                print("-- Falling back to the CPU (slower)")
                self.TORCH_DEVICE = "cpu"
                self.BATCH_SIZE = 1
                self.USE_HALF_PRECISION = False
        # team classifier device (cpu unless MPS_TEAM_ON_GPU)
        self.TEAM_DEVICE = self.TORCH_DEVICE if (self.TORCH_DEVICE != "mps" or MPS_TEAM_ON_GPU) else "cpu"
        # ultralytics 8.4 warns about the 'half' argument, so only pass it when it is switched on
        if self.TORCH_DEVICE != "cuda" and self.TORCH_DEVICE != "mps":
            self.USE_HALF_PRECISION = False   # FP16 only on the GPUs
        elif self.TORCH_DEVICE == "mps" and self.USE_HALF_PRECISION:
            print("-- FP16 (half precision) on the Apple GPU")
        self.HALF_KW = {"half": True} if self.USE_HALF_PRECISION else {}
        if self.HALF_KW:
            # ultralytics 8.4 logs "'half' is deprecated ... use 'quantize'" on every predict call (it still works): keep the log readable
            class _NoHalfWarning(logging.Filter):
                def filter(self, record):
                    return "'half' is deprecated" not in record.getMessage()
            logging.getLogger("ultralytics").addFilter(_NoHalfWarning())
        if (_players_coreml or _ball_coreml) and self.BATCH_SIZE != 1:
            # Core ML predicts one image per call, so keep batches at 1 (the Apple setting already is)
            print("-- Core ML in use: batch size forced to 1")
            self.BATCH_SIZE = 1

        #- ByteTrack: remember a player who disappears (hidden behind another player) for ~3 s instead of
        #  the 1 s default, and tell it the real detection frame rate so that time is right. Whatever
        #  ids still break are repaired afterwards by track_stitcher (player_id).
        try:
            self.BYTE_TRACKER = sv.ByteTrack(
                lost_track_buffer=90,   # supervision scales this by frame_rate/30 -> 90 = 3 seconds at any frame rate
                frame_rate=int(self.DETECTION_FPS))
        except TypeError:
            self.BYTE_TRACKER = sv.ByteTrack()   # older supervision with different argument names
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
            is_inside_pitch BOOL,
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
    @contextmanager
    def _stage(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._tick(name, t0)

    def _timed_iter(self, iterable, name):
        """yield from iterable, adding the time spent waiting for each item to the stage total"""
        it = iter(iterable)
        while True:
            t0 = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                return
            self._tick(name, t0)
            yield item

    def _tick(self, name, t0, background=False):
        """add the time since t0 to this stage's running total (timing only, no effect on results).
        background=True: work done in the background reader thread, overlapped with the main loop (not part of the sum)"""
        target = self.TIMINGS_BG if background else self.TIMINGS
        target[name] = target.get(name, 0.0) + (time.perf_counter() - t0)

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
                with self._stage('database writes'), sqlite3.connect(self.DATABASE_FILE) as conn:
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
        _t0 = time.perf_counter()
        if self.TORCH_DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                player_detections_batch = self.PLAYERS_PREDICTOR.predict(
                    frames_batch,
                    conf=0.6,
                    device=self.TORCH_DEVICE,
                    **self.HALF_KW,
                    imgsz=(self.PLAYER_MODEL_INPUT_SIZE['height'],self.PLAYER_MODEL_INPUT_SIZE['width']),
                    verbose=False
                )
        else:
            player_detections_batch = self.PLAYERS_PREDICTOR.predict(
                frames_batch,
                conf=0.6,
                device=self.TORCH_DEVICE,
                **self.HALF_KW,
                imgsz=(self.PLAYER_MODEL_INPUT_SIZE['height'],self.PLAYER_MODEL_INPUT_SIZE['width']),
                verbose=False
            )
        self._tick('player YOLO (1920x1920)', _t0)

        db_entries = []
        # loop each detection and prepare for db writing of detections:
        if len(player_detections_batch) > 0:

            for frame_idx, detection_result in enumerate(player_detections_batch):
                
                # convert to supervision
                frame_detections_supervision = sv.Detections.from_ultralytics(detection_result)
                # Add Tracking of Objects
                _t0 = time.perf_counter()
                frame_detections_with_tracking = self.BYTE_TRACKER.update_with_detections(frame_detections_supervision)   
                self._tick('ByteTrack update', _t0)
                # TEAM CLASSIFIER, one batched call per frame (was one call per player). Same crops, same model.
                team_results = {}
                _t0 = time.perf_counter()
                _crops, _crop_index = [], []
                for i in range(len(frame_detections_with_tracking.xyxy)):
                    _x0, _y0, _x1, _y1 = frame_detections_with_tracking.xyxy[i].tolist()
                    if int(_y0) < int(_y1) and int(_x0) < int(_x1):
                        _crop = frames_batch[frame_idx][:, int(_y0):int(_y1), int(_x0):int(_x1)]
                        if _crop.shape[1] > 0 and _crop.shape[2] > 0:
                            _crops.append(ImageUtils.create_team_assigner_tensor(_crop,
                                                                                 self.TEAM_MODEL_INPUT_SIZE['width'],
                                                                                 self.TEAM_MODEL_INPUT_SIZE['height']))
                            _crop_index.append(i)
                if _crops:
                    _team_batch = self.TEAM_TRACKER.predict(
                        torch.cat(_crops, dim=0),
                        conf=0.6,
                        device=self.TEAM_DEVICE,
                        # (no half= here: the tiny team model was slower in FP16, 19.5 vs 11.7 ms/frame)
                        imgsz=(192,192),
                        verbose=False
                    )
                    team_results = dict(zip(_crop_index, _team_batch))
                self._tick('team classifier (batched per frame)', _t0)

                # Loop results and get info from them that we want to store in db: Iterate over each detection
                for i in range(len(frame_detections_with_tracking.xyxy)):
                    
                    xmin, ymin, xmax, ymax = frame_detections_with_tracking.xyxy[i].tolist()
                    confidence = float(frame_detections_with_tracking.confidence[i])
                    class_id = int(frame_detections_with_tracking.class_id[i])
                    tracker_id = int(frame_detections_with_tracking.tracker_id[i]) if frame_detections_with_tracking.tracker_id is not None else None
                    class_name = frame_detections_with_tracking.data['class_name'][i] if 'class_name' in frame_detections_with_tracking.data else 'none'

                    # DETERMINE THE TEAM CLASSIFIER:
                    team_name = None
                    team_color = None
                    # Apply the crop to image the detection is taken from using the calculated bounding box:
                    if int(ymin) < int(ymax) and int(xmin) < int(xmax):
                        
                        # Select the specific frame from batch and get the bbounding box as image (tensor)
                        frame_image = frames_batch[frame_idx]  # Shape: (channels, height, width)
                        bbox_image = frame_image[:, int(ymin):int(ymax), int(xmin):int(xmax)]
                            
                        if bbox_image.shape[1] > 0 and bbox_image.shape[2] > 0:  # Validate dimensions
                            
                            # team result comes from the batched call above
                            team_detection = team_results.get(i)
                            if team_detection is not None:
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

                    # CHECK IF THE OBJECT IS ON THE PITCH (WE IGNORE ANY OBJECT OUTSIDE THE PITCH):
                    #if not self.is_detection_on_pitch(xmin, ymin, xmax, ymax, self.PITCH_HOMOGRAPHY):
                    #    continue  # Skip this detection as it's not on the pitch
                    # Check if the detection is inside the pitch boundaries:
                    is_inside_pitch = self.is_detection_box_inside_pitch(frame_ids[frame_idx], class_name, xmin*scaler_x, ymin*scaler_y, xmax*scaler_x, ymax*scaler_y)


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
                        'is_inside_pitch': is_inside_pitch,
                        'image_path': ''
                    })
                    #print(class_name)
        else:
            print("⚠️ No player/referee/goalkeeper detections found in this batch.")
            return

        # ✅ Write tracked detections to the database in bulk
        with self._stage('database writes'), sqlite3.connect(self.DATABASE_FILE) as conn:
            cursor = conn.cursor()
            for det in db_entries:
                cursor.execute('''
                    INSERT INTO detected_objects (
                        frame_id, xmin, ymin, xmax, ymax, confidence, class_id, tracker_id, class_name,
                        x_transformed_metres, y_transformed_metres, speed_km_per_hour, total_distance_metres, is_inside_pitch,
                        team, team_color, image_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    det['frame_id'], det['xmin'], det['ymin'], det['xmax'], det['ymax'], det['confidence'], det['class_id'], det['tracker_id'],
                    det['class_name'], det['x_transformed_metres'], det['y_transformed_metres'], det['speed_km_per_hour'],
                    det['total_distance_metres'], det['is_inside_pitch'], det['team'], det['team_color'], det['image_path']
                ))
            conn.commit()


    ########################################## - BALL DETECTIONS:
            
    def process_ball_detections(self, frames_batch, frame_ids):
        
        """
            Perform batch ball detection, filter only 'ball' detections,
            and save results to the database efficiently while preserving aspect ratio.
        """

        # ✅ Batch YOLO detections
        _t0 = time.perf_counter()
        if self.TORCH_DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                ball_detections_batch = self.BALL_PREDICTOR.predict(
                    frames_batch,
                    conf=0.3,
                    device=self.TORCH_DEVICE,
                    **self.HALF_KW,
                    imgsz=(self.BALL_MODEL_INPUT_SIZE['height'],self.BALL_MODEL_INPUT_SIZE['width']),
                    verbose=False
                )
        else:
            ball_detections_batch = self.BALL_PREDICTOR.predict(
                frames_batch,
                conf=0.3,
                device=self.TORCH_DEVICE,
                **self.HALF_KW,
                imgsz=(self.BALL_MODEL_INPUT_SIZE['height'],self.BALL_MODEL_INPUT_SIZE['width']),
                verbose=False
            )
        self._tick('ball YOLO (1920x1920)', _t0)

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

                    # scale the bounding boxes back to the original video size:
                    scaler_x = self.ORIGINAL_VIDEO_WIDTH/self.PLAYER_MODEL_INPUT_SIZE['width']
                    scaler_y = self.ORIGINAL_VIDEO_HEIGHT/self.PLAYER_MODEL_INPUT_SIZE['height']
                    
                    # Check if the detection is inside the pitch boundaries:
                    is_inside_pitch = self.is_detection_box_inside_pitch(frame_ids[frame_idx], class_name, xmin*scaler_x, ymin*scaler_y, xmax*scaler_x, ymax*scaler_y)

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
                        'is_inside_pitch': is_inside_pitch,
                        'image_path': ''
                    })
        
        else:
            #print("⚠️ No 'ball' detections found in this batch.")
            return

        # ✅ Write tracked detections to the database in bulk
        with self._stage('database writes'), sqlite3.connect(self.DATABASE_FILE) as conn:
            cursor = conn.cursor()
            for det in db_entries:
                if class_name == 'ball':
                    cursor.execute('''
                        INSERT INTO detected_objects (
                                frame_id, xmin, ymin, xmax, ymax, confidence, class_id, tracker_id, class_name,
                                x_transformed_metres, y_transformed_metres, speed_km_per_hour, total_distance_metres, is_inside_pitch, 
                                team, team_color, image_path
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        det['frame_id'], det['xmin'], det['ymin'], det['xmax'], det['ymax'], det['confidence'], det['class_id'], det['tracker_id'],
                        det['class_name'], det['x_transformed_metres'], det['y_transformed_metres'], det['speed_km_per_hour'],
                        det['total_distance_metres'], det['is_inside_pitch'], det['team'], det['team_color'], det['image_path']
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
            # half time is optional (a short test clip has none)
            ht_start = str(game_record.get('half_time_start_seconds') or '').strip()
            ht_end = str(game_record.get('half_time_end_seconds') or '').strip()
            if ht_start and ht_end:
                exclusion_ranges.append((self.time_to_seconds(ht_start), self.time_to_seconds(ht_end)))
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

        # Stop reading at the game end: everything after it is excluded anyway, but the loop would still walk
        # through (and decode) every remaining chunk of the video just to skip each frame.
        try:
            _game_end_frame = int(math.ceil(self.time_to_seconds(self.GAME_RECORD['game_end_seconds']) * original_fps))
            if 0 < _game_end_frame < end_frame:
                print(f"🏁 Game ends at frame {_game_end_frame}: not reading the {end_frame - _game_end_frame} frames after it")
                end_frame = _game_end_frame
                total_chunks = (end_frame + chunk_frames - 1) // chunk_frames
        except Exception as e:
            print(f"⚠️ Could not apply game end to the frame range ({type(e).__name__}: {e}) - reading to the end")

        # GET VIDEO FRAMES: Get the frames for the chunk to process and prepare them (resize/tensor create):
        def read_chunk(chunk_start_frame):
            """read + prepare one chunk of frames. Runs in a background thread so the next chunk is being
            decoded and resized while the models work on the current one."""
            chunk_end_frame = min(chunk_start_frame + chunk_frames, end_frame)
            frames = []
            frame_ids = []
            _t_read = time.perf_counter()

            _t_seek = time.perf_counter()
            # Seeking (cap.set) makes the decoder restart from the nearest key frame and decode forward, which can cost
            # seconds. Chunks follow on from each other, so only seek when the reader is not already at the chunk start.
            if reader_state['pos'] != chunk_start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start_frame)
            reader_state['pos'] = None     # unknown until this chunk has been read to its end
            _seek_s, _decode_s, _convert_s = time.perf_counter() - _t_seek, 0.0, 0.0
            current_frame_id = chunk_start_frame
            ret = True

            # loop thro each frome in the video in the chunk and skip where necessary due to detection fps or exclusion ranges:
            while current_frame_id < chunk_end_frame:
                
                #print(f"current_frame_id: {current_frame_id}")
                _t_dec = time.perf_counter()
                ret, frame = cap.read()
                _decode_s += time.perf_counter() - _t_dec
                
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
                _t_conv = time.perf_counter()
                frame_tensor = ImageUtils.create_tensor_and_resize(frame,self.PLAYER_MODEL_INPUT_SIZE['width'],
                                             self.PLAYER_MODEL_INPUT_SIZE['height'],
                                             self.ORIGINAL_VIDEO_WIDTH,
                                             self.ORIGINAL_VIDEO_HEIGHT
                                             )

                _convert_s += time.perf_counter() - _t_conv
                # ADD TO ARRAY/DICT we will process for this chunck
                frames.append(frame_tensor)
                frame_ids.append(current_frame_id)
                #print(f"✅ Adding Frame ID: {current_frame_id}")

                # LOOP: update to the new frame id and rerun: # Checking if we should skip the frame due to FPS
                current_frame_id = self.skip_frames_due_to_fps(cap, current_frame_id, frame_skip)

            if ret:
                reader_state['pos'] = current_frame_id   # the capture sits at the next unread frame = start of the next chunk
            self._tick('video read + resize to tensor', _t_read, background=True)
            read_secs[chunk_start_frame] = (time.perf_counter() - _t_read, len(frames), _seek_s, _decode_s, _convert_s)
            return frames, frame_ids

        chunk_starts = list(range(0, end_frame, chunk_frames))
        reader_state = {'pos': None}   # frame index the video capture is positioned at (None = unknown)
        wait_lines = []  # one line per chunk, printed again at the end of the run
        read_secs = {}   # chunk start frame -> (seconds the background read took, frames read)  [diagnostic]
        reader_pool = ThreadPoolExecutor(max_workers=1)
        pending_read = reader_pool.submit(read_chunk, chunk_starts[0]) if chunk_starts else None

        for chunk_idx, chunk_start_frame in enumerate(chunk_starts, start=1):
            
            chunk_end_frame = min(chunk_start_frame + chunk_frames, end_frame)
            print(f"\n🗂️ -- Processing video chunk (video frames to tensors) -- Video time: {frame_to_timestamp(chunk_start_frame,self.ORIGINAL_VIDEO_FPS)} - {frame_to_timestamp(chunk_end_frame,self.ORIGINAL_VIDEO_FPS)} | Chuncks: {chunk_idx} of {total_chunks} | Frames {chunk_start_frame} - {chunk_end_frame}")

            _t_wait = time.perf_counter()
            frames, frame_ids = pending_read.result()      # normally ready already: it was read during the previous chunk
            self._tick('waiting for frames from the background reader', _t_wait)
            _rs = read_secs.get(chunk_start_frame, (0, 0, 0, 0, 0))
            _line = (f"chunk {chunk_idx:>3}: waited {time.perf_counter() - _t_wait:5.2f}s | background read {_rs[0]:5.2f}s for {_rs[1]} frames "
                     f"(seek {_rs[2]:.2f}s, decode {_rs[3]:.2f}s, resize+tensor {_rs[4]:.2f}s)")
            wait_lines.append(_line)
            print("   ⏳ " + _line)
            if chunk_idx < len(chunk_starts):
                pending_read = reader_pool.submit(read_chunk, chunk_starts[chunk_idx])   # start reading the next chunk now
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
                num_workers=0,   # frames are already in memory: worker processes only add start-up + copy cost (esp. on macOS)
                pin_memory=self.TORCH_DEVICE == "cuda"
            )

            # PROCESS DETECTIONS: loop thro each chunk processing 1 batch of frames at a time:
            print(f"🔄 Starting Detections for Chunk of video frames {chunk_idx} of {total_chunks}... (chunk size= {chunk_duration} seconds)")
            with tqdm.tqdm(total=len(frames), desc=f"🔍 Processing Chunk {chunk_idx}/{total_chunks}", unit="frame") as pbar:
                try:

                    # BYTE TRACKER: Restore state before processing the next batch (if needed)
                    with open('../output/tracker_state.pkl', 'rb') as f:
                        state = pickle.load(f)
                        self.BYTE_TRACKER.tracked_tracks = state['tracks']
                        self.BYTE_TRACKER.lost_tracks = state['lost_tracks']

                    # LOOP: loop the batch and do the detections:
                    for batch_idx, batch_frames in enumerate(self._timed_iter(dataloader, 'data loader wait')):
                        
                        # Move batch to the selected device
                        # (CUDA only: on MPS the frames stay on the CPU - crops/pitch code use them as normal tensors,
                        #  and ultralytics copies each batch to the GPU itself inside predict())
                        batch_frames = batch_frames.to(self.TORCH_DEVICE if self.TORCH_DEVICE == "cuda" else "cpu")
                        if self.USE_HALF_PRECISION and self.TORCH_DEVICE == "cuda":
                            batch_frames = batch_frames.half()  # Use FP16 on CUDA if enabled

                        start_idx = batch_idx * self.BATCH_SIZE
                        end_idx = start_idx + len(batch_frames)

                        # PLAYHEAD POSITION FOR REFERENCE: in min:sec
                        current_frame_id = frame_ids[start_idx]
                        current_time = frame_to_timestamp(current_frame_id,self.ORIGINAL_VIDEO_FPS)
                        pbar.set_description(f"🔍 Processing Detections at video time: {current_time}")

                        # PROCESS: process the detections and save them to db:
                        self.process_pitch_detections(batch_frames, frame_ids[start_idx:end_idx])
                        self.process_player_detections(batch_frames, frame_ids[start_idx:end_idx])
                        self.process_ball_detections(batch_frames, frame_ids[start_idx:end_idx])

                        self.TIMED_FRAMES += len(batch_frames)
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

        print("\n⏳ Background reader, per chunk (first 12 chunks):")
        for _l in wait_lines[:12]:
            print("   " + _l)
        reader_pool.shutdown(wait=True)
        cap.release()
        

        # TIME FOR FULL PROCESS: Calculate elapsed time for entire detections process:
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time
        minutes, seconds = divmod(int(total_time), 60)
        print(f"✅ Processing time: {minutes}:{seconds} sec")

        # WHERE THE TIME WENT (baseline for speed work)
        n = max(1, self.TIMED_FRAMES)
        accounted = sum(self.TIMINGS.values())
        rows = sorted(self.TIMINGS.items(), key=lambda kv: -kv[1]) + [('everything else (db writes, loader, python)', max(0.0, total_time - accounted))]
        print(f"⏱️ Time per stage over {n} frames ({total_time / n * 1000:.0f} ms/frame overall, {n / total_time:.2f} frames/s):")
        for name, sec in rows:
            print(f"   {name:<46}{sec:8.1f} s  {100 * sec / total_time:5.1f}%  {sec / n * 1000:7.1f} ms/frame")
        for name, sec in self.TIMINGS_BG.items():
            print(f"   (background, overlapped) {name:<30}{sec:6.1f} s  {sec / n * 1000:7.1f} ms/frame")


