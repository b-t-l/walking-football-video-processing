import numpy as np 
import cv2
import sqlite3
import json
from tqdm import tqdm
from utils import *
from data import GetGameData
from .fitted_pitch_transformer import FittedPitchTransformer
import time

class CreateViewTransformations():

    # CREATE OUR VIEW TRANSFORMER:
    def __init__(self,DATABASE_FILE,GAME_RECORD,PITCH_DETECTION_OVERRIDE):

        print("-------------------------------------------------")
        print("CREATE VIEW TRANSFORMATIONS: from db update to add new positions from homography")

        self.GAME_RECORD = GAME_RECORD
        self.DATABASE_FILE = DATABASE_FILE
        self.PITCH_DETECTION_OVERRIDE = PITCH_DETECTION_OVERRIDE
        self.PITCH_OVERRIDE_KEYPOINTS = StringUtils.string_vertices_to_nparray(GAME_RECORD,0)

        #- Dimensions of our whole pitch in metres (legacy fallback only, see below)
        court_width = 21
        court_length = 46

        #- what should these point be in a top view ? (legacy fallback only, see below)
        self.OVERRIDE_TARGET_VERTICES = np.array([
            [0, 0],
            [0, court_width],
            [court_length, court_width],
            [court_length, 0]
        ])

        #- PREFERRED PATH: a per-video Pitch Calibrator export (pitch_calibrator_json column).
        #  This fits a radial-distortion + homography model from the full calibration point
        #  set (outline + centre circle + goal-end arcs), fresh for this specific video, to
        #  correct for the residual lens curvature a flat homography can't represent. See
        #  view_transformer/fitted_pitch_transformer.py for the full approach and caveats.
        calibrator_json = GAME_RECORD.get('pitch_calibrator_json') if hasattr(GAME_RECORD, 'get') else None
        if isinstance(calibrator_json, str) and calibrator_json.strip():
            self.USE_FITTED_TRANSFORMER = True
            self.fitted_transformer = FittedPitchTransformer(json.loads(calibrator_json))
        else:
            #- LEGACY FALLBACK: no calibrator JSON for this game yet, keep the old flat
            #  4-corner homography so older games-logger.xlsx rows keep working unchanged.
            self.USE_FITTED_TRANSFORMER = False
            print("-------------------------------------------------")
            print("WARNING: no pitch_calibrator_json found for this game - falling back to the "
                  "legacy flat cv2.getPerspectiveTransform (no lens-curvature correction).")
            left_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_bottom'].split(',')))
            left_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_top'].split(',')))
            right_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_top'].split(',')))
            right_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_bottom'].split(',')))
            self.OVERRIDE_SOURCE_VERTICES = np.array([
                 left_bottom_tuple,
                 left_top_tuple,
                 right_top_tuple,
                 right_bottom_tuple
            ])
            pitch_vertices = self.OVERRIDE_SOURCE_VERTICES.astype(np.float32)
            target_vertices = self.OVERRIDE_TARGET_VERTICES.astype(np.float32)
            self.OVERRIDE_PERSPECTIVE_TRANSFORMER = cv2.getPerspectiveTransform(pitch_vertices, target_vertices)


    ########## IS POINT ON LEFT OR RIGHT SIDE OF THE PITCHdetermine if a point is left or right of the line running from centre_top to centre_bottom.
    def is_point_left_or_right(self,pitch_keypoints, obj_x, obj_y):
        """
        Determine if a point (obj_x, obj_y) is left or right of the line 
        running from centre_top to centre_bottom.
        
        Args:
            pitch_keypoints (list): List of keypoints with 'name', 'x', 'y'.
            obj_x (int): X-coordinate of the object.
            obj_y (int): Y-coordinate of the object.
        
        Returns:
            str: 'left', 'right', or 'on the line'
        """
        # Extract centre_top and centre_bottom from the keypoints
        centre_top = next(kp for kp in pitch_keypoints if kp['name'] == 'centre_top')
        centre_bottom = next(kp for kp in pitch_keypoints if kp['name'] == 'centre_bottom')
        
        x1, y1 = centre_top['x'], centre_top['y']
        x2, y2 = centre_bottom['x'], centre_bottom['y']
        
        # Calculate cross product
        cross_product = (x2 - x1) * (obj_y - y1) - (y2 - y1) * (obj_x - x1)
        
        if cross_product > 0:
            return 'left'
        elif cross_product < 0:
            return 'right'
        else:
            return 'centre'
        
    '''
    # TRANSFORM A BOUNDING BOX TO POSITION IWTH HOMOGRAPHY:
    def transform_box(self, class_name, tracker_id, box, frame_size, pitch_keypoints_final, pitch_vertices):

        p = (int(box[2]),int(box[2]))
        x1 = box[0]
        x2 = box[2]
        y1 = box[1]
        y2 = box[3]
        frame_width , frame_height = frame_size

        #print(f"{frame_width} x {frame_height}")
        #print(box)
        #print(pitch_keypoints_final)

        # work out if the object is on the left/right of the line down the centre of the pitch:
        object_pitch_position = self.is_point_left_or_right(pitch_keypoints_final, ((x2-x1)/2 + x1), ((y2-y1)/2 + y1))

        # pick the correct foot position based on the results ( left side of pitch foot position = bottom right etc )
        if class_name == 'ball':
            # ball is always centre
            p = (int(x2),int(y2))
        else:
            if object_pitch_position == 'centre':
                p = (int(x2),int(y2))
            if object_pitch_position == 'left':
                p = (int(x2),int(y2))
            if object_pitch_position == 'right':
                p = (int(x1),int(y2))

        # check if the point on the field is inside the field:
        #is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        #if not is_inside:
        #    return None

        #reshaped_box = box.reshape(-1,1,2).astype(np.float32)
        reshaped_box = np.array([[[p[0], p[1]]]], dtype=np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_box,self.persepctive_trasnformer)

        return tranform_point.reshape(-1,2)
    '''

    ########## TRANSFORM A POINT: TO POSITION WITH HOMOGRAPHY:
    def transform_point(self, point):

        # Define point as a tuple (x, y)
        x, y = point

        # Reshape point for perspective transform
        reshaped_point = np.array([[[x, y]]], dtype=np.float32)

        # Apply the perspective transform
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_trasnformer)

        # Return transformed point as integer values (x, y)
        transformed_x, transformed_y = transformed_point[0][0]
        return (int(transformed_x), int(transformed_y))

    '''
    ############################################### GENERAL REUSED FUNCTIONS
    def get_center_of_bbox(x1,y1,x2,y2):
        return int((x1+x2)/2),int((y1+y2)/2)
    '''


    ########## TRANSFORM BOUNDING BOX: work out the point to use from the bounding box and transform it using the homography:
    def transform_object(self,frame_id,xmin, ymin, xmax, ymax, class_id, class_name, tracker_id, pitch_vertices):
        
        """
        Calculate transformed coordinates based on pitch vertices.
        if we override then we are using set vertices for each corner of the pitch and not the ones detected in the detections.
        """
        
        # POSITION: calculate the point we will use from the bounding box to transform: differs for player/ball etc
        point_x = xmax
        point_y = ymax

        # work out if the object is on the left/right of the line down the centre of the pitch:
        if self.PITCH_DETECTION_OVERRIDE == True:
            # overide them if we are using PITCH_CORRECTION_OVERDIDE from the gamelogger:
            object_pitch_position = self.is_point_left_or_right(self.PITCH_OVERRIDE_KEYPOINTS, ((xmax-xmin)/2 + xmin), ((ymax-ymin)/2 + ymin))
        else:
            object_pitch_position = self.is_point_left_or_right(pitch_vertices, ((xmax-xmin)/2 + xmin), ((ymax-ymin)/2 + ymin))
        
        # DETERMINE THE RIGHT X AND Y POINTS TO USE BASED ON LEFT RIGHT SIDE OF PITCH:
        if class_name == 'ball':
            # ball is always the centre bottom of the bounding box:
            point_x = int((xmin+xmax)/2)
        else:
            if object_pitch_position == 'centre':
                point_x = int((xmin+xmax)/2)
            if object_pitch_position == 'left':
                point_x = int(xmax)
            if object_pitch_position == 'right':
                point_x = int(xmin)

        # TRANSFORM: 
        if self.PITCH_DETECTION_OVERRIDE == True:
            if self.USE_FITTED_TRANSFORMER:
                # fitted radial-distortion + homography model, fresh-fit for this video
                x_transformed, y_transformed = self.fitted_transformer.transform_point(point_x, point_y)
            else:
                # legacy: flat perspective transformer from the gamelogger excel (constant)
                reshaped_point = np.array([[[point_x,point_y]]], dtype=np.float32)
                transform_point = cv2.perspectiveTransform(reshaped_point,self.OVERRIDE_PERSPECTIVE_TRANSFORMER)
                x_transformed, y_transformed = float(transform_point[0][0][0]), float(transform_point[0][0][1])

            #if tracker_id == 1:
                #print(f"Tracker_id: {tracker_id} : Frame_id: {frame_id} : {xmin}, {ymin}, {xmax}, {ymax} : position: {object_pitch_position} : point used: {point_x},{point_y} : transformed: {x_transformed}, {y_transformed}")
        else:
            # create a new perspective transformer as the pitch detection change for each frame:
            pitch_vertices = pitch_vertices.astype(np.float32)
            #print(pitch_vertices)
            target_vertices = self.OVERRIDE_TARGET_VERTICES.astype(np.float32)
            PITCH_DETECTION_PERSPECTIVE_TRANSFORMER = cv2.getPerspectiveTransform(pitch_vertices, target_vertices)
            reshaped_point = np.array([[[point_x,point_y]]], dtype=np.float32)
            transform_point = cv2.perspectiveTransform(reshaped_point,PITCH_DETECTION_PERSPECTIVE_TRANSFORMER)
            x_transformed, y_transformed = float(transform_point[0][0][0]), float(transform_point[0][0][1])

        # calculate the transformed position
        return x_transformed, y_transformed


    ########## PITCH TRANSFORMATION: convert a pitch vertice to a transformed vertice using the homography:
    def custom_pitch_transformation(self, x, y):
        """
        Custom transformation for detected_pitch records.
        Placeholder logic - Replace with actual transformation.
        """
        if self.USE_FITTED_TRANSFORMER:
            # fitted radial-distortion + homography model, fresh-fit for this video
            x_transformed, y_transformed = self.fitted_transformer.transform_point(x, y)
        else:
            # legacy: flat perspective transformer from the gamelogger excel (constant)
            reshaped_point = np.array([[[x,y]]], dtype=np.float32)
            transform_point = cv2.perspectiveTransform(reshaped_point,self.OVERRIDE_PERSPECTIVE_TRANSFORMER)
            x_transformed, y_transformed = float(transform_point[0][0][0]), float(transform_point[0][0][1])

        return x_transformed, y_transformed


    ############################################### RUN
    def run(self, batch_size=1000):

        start_time = time.time()  # Start the timer for overall process time

        conn = sqlite3.connect(self.DATABASE_FILE)
        cursor = conn.cursor()
            
        try:
            # Step 1: Fetch unique frame_ids from detected_pitch
            cursor.execute('SELECT DISTINCT frame_id FROM detected_pitch')
            frame_ids = [row[0] for row in cursor.fetchall()]
            #print(f"📊 Found {len(frame_ids)} unique frame_ids in detected_pitch.")

            # Step 2: Process frame_ids in batches
            for i in tqdm(range(0, len(frame_ids), batch_size), desc="🔄 Processing Frames"):
                batch_frame_ids = frame_ids[i:i + batch_size]
                #print(i)

                # Fetch pitch detections for current batch
                cursor.execute(f'''
                    SELECT _id, frame_id, x, y 
                    FROM detected_pitch 
                    WHERE frame_id IN ({','.join('?' * len(batch_frame_ids))})
                ''', batch_frame_ids)
                pitch_rows = cursor.fetchall()

                ######## Group pitch data by frame_id and process each record
                pitch_data = {}
                for _id, frame_id, x, y in pitch_rows:

                    # Pass through the custom pitch transformation function
                    x_transformed, y_transformed = self.custom_pitch_transformation(x, y)

                    # Update transformed coordinates back into the database
                    cursor.execute('''
                        UPDATE detected_pitch 
                        SET x_transformed_metres = ?, y_transformed_metres = ? 
                        WHERE _id = ?
                    ''', (x_transformed, y_transformed, _id))

                    if frame_id not in pitch_data:
                        pitch_data[frame_id] = []
                    pitch_data[frame_id].append((x, y, x_transformed, y_transformed))

                ####### Process player detections for each frame_id
                for frame_id in batch_frame_ids:
                    if frame_id not in pitch_data:
                        continue  # Skip if no pitch data for this frame_id
                        
                    pitch_vertices = np.array([(x, y) for x, y, _, _ in pitch_data[frame_id]])  # Convert to NumPy array

                    # Fetch player detections for current frame_id
                    cursor.execute('''
                        SELECT _id, xmin, ymin, xmax, ymax, class_id, class_name, tracker_id
                        FROM detected_objects 
                        WHERE frame_id = ?
                    ''', (frame_id,))
                    player_rows = cursor.fetchall()

                    # Update each player record
                    for _id, xmin, ymin, xmax, ymax, class_id, class_name, tracker_id in player_rows:

                        # Example: Use center point for transformation
                        x_trans, y_trans = self.transform_object(frame_id,
                                                                 xmin, 
                                                                 ymin, 
                                                                 xmax, 
                                                                 ymax, 
                                                                 class_id, 
                                                                 class_name,
                                                                 tracker_id, 
                                                                 pitch_vertices)

                        cursor.execute('''
                            UPDATE detected_objects 
                            SET x_transformed_metres = ?, y_transformed_metres = ? 
                            WHERE _id = ?
                        ''', (x_trans, y_trans, _id))

                # Commit batch updates
                conn.commit()
                    
        except sqlite3.Error as e:
            print(f"⚠️ SQLite Error: {e}")
        finally:
            
            cursor.close()
            conn.close()
            print("🔒 Database connection closed.")

            # TIME FOR FULL PROCESS: Calculate elapsed time for entire process:
            end_time = time.time()  # Stop the timer
            total_time = end_time - start_time
            minutes, seconds = divmod(int(total_time), 60)
            print(f"✅ Processing time: {minutes}:{seconds} sec")
