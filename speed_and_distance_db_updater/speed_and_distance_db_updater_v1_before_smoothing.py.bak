import sqlite3
import math
from tqdm import tqdm
import time

class SpeedDistanceUpdater:

    ########## INITIALIZE THE SPEED/DISTANCE CALCULATOR:
    def __init__(self, DATABASE_FILE, GAME_RECORD, DURATION, batch_size=10000):

        print("-------------------------------------------------")
        print("CALCULATE SPEED/DISTANCE: from db update each object calculating speed/distance")

        """
        Initialize with database file, game record, batch size, and optional duration filter.
        duration: Tuple (start_frame, end_frame) to limit processing.
        """
        self.batch_size = batch_size
        self.GAME_RECORD = GAME_RECORD
        self.DATABASE_FILE = DATABASE_FILE
        self.conn = sqlite3.connect(self.DATABASE_FILE)
        self.cursor = self.conn.cursor()
        

        # Fetch game details
        self.cursor.execute('''
            SELECT _id, game_id, video, video_fps, detection_fps
            FROM game
            WHERE game_id = ?
        ''', (self.GAME_RECORD['game_id'],))
        self.DB_GAME_DETAILS = self.cursor.fetchone()
        self.DETECTION_FPS = self.DB_GAME_DETAILS[4]
        self.duration = (0,DURATION*self.DETECTION_FPS)  # Store the duration filter


    ########## RUN THE SPEED/DISTANCE CALCULATOR:
    def run(self):

        start_time = time.time()  # Start the timer for overall process time

        """Fetch, calculate, and update database records in batches."""
        self.cursor.execute('''
            SELECT DISTINCT tracker_id FROM detected_objects
            WHERE class_name = "player"
        ''')
        tracker_ids = self.cursor.fetchall()

        processed_records = 0

        with tqdm(total=len(tracker_ids), desc="🔄 Processing each player via Tracker ID", unit="tracker") as tracker_pbar:
            for tracker_id in tracker_ids:
                tracker_id = tracker_id[0]

                self.cursor.execute('''
                        SELECT frame_id, x_transformed_metres, y_transformed_metres, speed_km_per_hour, total_distance_metres, is_inside_pitch, xmin, ymin, xmax, ymax
                        FROM detected_objects 
                        WHERE tracker_id = ? AND class_name = "player"
                        ORDER BY frame_id
                    ''', (tracker_id,))
                
                # Build the SQL query dynamically based on DURATION
                #if self.duration:
                #    start_frame, end_frame = self.duration
                #    self.cursor.execute('''
                #        SELECT frame_id, x_transformed_metres, y_transformed_metres, speed_km_per_hour, total_distance_metres 
                #        FROM detected_objects 
                #        WHERE tracker_id = ? AND class_name = "player" 
                #        AND frame_id BETWEEN ? AND ?
                #        ORDER BY frame_id
                #    ''', (tracker_id, start_frame, end_frame))
                #else:
                #    self.cursor.execute('''
                #        SELECT frame_id, x_transformed_metres, y_transformed_metres, speed_km_per_hour, total_distance_metres 
                #        FROM detected_objects 
                #        WHERE tracker_id = ? AND class_name = "player"
                #        ORDER BY frame_id
                #    ''', (tracker_id,))

                records = self.cursor.fetchall()

                # Convert records to dictionaries
                records = [
                    {
                        'frame_id': row[0],
                        'x_transformed_metres': row[1],
                        'y_transformed_metres': row[2],
                        'speed_km_per_hour': row[3],
                        'total_distance_metres': row[4],
                        'is_inside_pitch': row[5],
                        'bbox': (row[6],row[7],row[8],row[9]),
                    }
                    for row in records
                ]

                # Initialize variables
                accumulated_distance = 0
                updates_batch = []

                # Define a threshold for maximum reasonable distance change
                max_distance_change = 5.0  # Adjust this value based on your data
                max_bounding_box_area_change_between_frames = 0.3  

                # Perform calculations
                for i in range(len(records)):
                    if i == 0:
                        # First record: No previous frame to compare
                        speed = 0.0
                        distance = 0.0
                        prev_bbox_area = None  # Initialize previous bounding box area
                    else:
                        x1, y1 = records[i - 1]['x_transformed_metres'], records[i - 1]['y_transformed_metres']
                        x2, y2 = records[i]['x_transformed_metres'], records[i]['y_transformed_metres']
                        bbox = records[i]['bbox']

                        if None in (x1, y1, x2, y2):
                            speed = 0.0
                            distance = 0.0
                        else:
                            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                            # Check if the distance change is reasonable
                            if distance > max_distance_change:
                                distance = 0.0  # Ignore this change
                            else:
                                accumulated_distance += distance

                            frame_diff = records[i]['frame_id'] - records[i - 1]['frame_id']
                            time_diff = frame_diff / self.DETECTION_FPS if frame_diff > 0 else 1

                            speed = (distance / time_diff) * 3.6 if time_diff > 0 else 0.0

                        # Set speed to 0 if the player is outside the pitch
                        if records[i]['is_inside_pitch'] == 0:
                            speed = 0.0

                        # Set the speed to 0 if the bounding box has increased or decreased in area by more than 40% between frames
                        # Calculate the area of the current bounding box
                        current_bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        # Check if the bounding box area has changed by more than 40%
                        if prev_bbox_area is not None:
                            area_change = abs(current_bbox_area - prev_bbox_area) / prev_bbox_area
                            if area_change > max_bounding_box_area_change_between_frames:
                                speed = 0.0

                        # Update the previous bounding box area
                        prev_bbox_area = current_bbox_area

                    # if the values are too high we know there was a mistake somewhere in the transform so we simply assign zero:
                    #if speed >= 40:

                        #print(f"- Excessive speed detected: Tracker_id: {tracker_id} : Frame_id: {records[i]['frame_id']} : previous frame x,y: {x1}, {y1} : current frame x,y: {x2}, {y2} : frame_diff: {frame_diff} : time_diff: {time_diff} : distance: {distance} : speed: {speed}")
                        #print(f"* Excessive speed detected: Tracker_id: {tracker_id} : Frame_id: {records[i]['frame_id']} : speed: {speed}")
                        #speed = 0.0
                        #distance = 0.0
                    
                    updates_batch.append((speed, accumulated_distance, records[i]['frame_id'], tracker_id))
                    processed_records += 1

                    if len(updates_batch) >= self.batch_size:
                        self._execute_batch_update(updates_batch)
                        updates_batch = []

                if updates_batch:
                    self._execute_batch_update(updates_batch)

                tracker_pbar.update(1)

        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        print("🔒 Database connection closed.")

        # TIME FOR FULL PROCESS: Calculate elapsed time for entire process:
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time
        minutes, seconds = divmod(int(total_time), 60)
        print(f"✅ Processing time: {minutes}:{seconds} sec")

    ########## EXECUTE BATCH UPDATES:   
    def _execute_batch_update(self, updates_batch):
        """Execute batched updates."""
        self.cursor.executemany('''
            UPDATE detected_objects
            SET speed_km_per_hour = ?, total_distance_metres = ?
            WHERE frame_id = ? AND tracker_id = ?
        ''', updates_batch)
        self.conn.commit()
