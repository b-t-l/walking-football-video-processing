import sqlite3
import math
from tqdm import tqdm
import time

class BallPosessionUpdater:
    
    def __init__(self, db_path, GAME_RECORD):

        print("-------------------------------------------------")
        print("CALCULATE BALL POSSESSION: from db update each object calculating if in posession of ball")

        self.db_path = db_path
        self.batch_size = 10000
        self.conn = None
        self.cursor = None
        self.DISTANCE_TOLERANCE = 1.5 # proximty in metres that we use to determine in the player is in posession of the ball

    def connect(self):
        """Establish a connection to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def disconnect(self):
        """Close the database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def calculate_ball_posession(self, records):
        """Calculate ball possession for a single frame.
            it can only be assigned to a single player and this is the closest player to the ball
        """
        # Extract ball position
        ball_position = next(
            (record for record in records if record['class_name'] == 'ball'), 
            None
        )
        if not ball_position:
            return []  # No ball found, skip possession calculation

        ball_x, ball_y = ball_position['x_transformed_metres'], ball_position['y_transformed_metres']
        distance_tolerance = self.DISTANCE_TOLERANCE  # Define distance threshold in metres

        # Calculate distances and determine possession
        updated_records = []
        closest_player = None
        closest_distance = float('inf')  # Initialize with infinity

        for record in records:
            if record['class_name'] in ['player', 'goalkeeper']:
                player_x, player_y = record['x_transformed_metres'], record['y_transformed_metres']
                distance = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

                # Check if the player is within the distance tolerance
                if distance <= distance_tolerance:
                    # Determine if this player is the closest to the ball
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_player = record  # Keep track of the closest player

        # If a closest player is found, assign possession
        if closest_player:
            closest_player['ball_posession'] = True
            updated_records.append(closest_player)

        # Mark all other players as not in possession
        for record in records:
            if record['class_name'] in ['player', 'goalkeeper'] and record != closest_player:
                record['ball_posession'] = False
                updated_records.append(record)

        return updated_records
    
    '''
    def calculate_ball_posession(self, records):
        """Calculate ball possession for a single frame."""
        # Extract ball position
        ball_position = next(
            (record for record in records if record['class_name'] == 'ball'), 
            None
        )
        #print(ball_position)
        if not ball_position:
            return []  # No ball found, skip possession calculation

        ball_x, ball_y = ball_position['x_transformed_metres'], ball_position['y_transformed_metres']
        distance_tolerance = self.DISTANCE_TOLERANCE  # Define distance threshold in metres

        # Calculate distances and determine possession
        updated_records = []
        for record in records:
            if record['class_name'] in ['player', 'goalkeeper']:
                player_x, player_y = record['x_transformed_metres'], record['y_transformed_metres']
                distance = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)
                #print(distance)
                # Assign possession
                record['ball_posession'] = distance <= distance_tolerance
                updated_records.append(record)

        return updated_records
    '''

    def run(self):

        start_time = time.time()  # Start the timer for overall process time
        """Establish a connection to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        """Fetch, calculate, and update database records in batches."""
        self.cursor.execute('''
            SELECT DISTINCT frame_id FROM detected_objects
            WHERE class_name IN ("player", "goalkeeper", "ball")
        ''')
        frame_ids = [row[0] for row in self.cursor.fetchall()]

        processed_records = 0  # Counter for tracking progress

        with tqdm(total=len(frame_ids), desc="Processing Frames", unit="frame_id") as tracker_pbar:
            for frame_id in frame_ids:
                # Fetch records for this frame_id
                self.cursor.execute('''
                    SELECT frame_id, class_name, x_transformed_metres, y_transformed_metres, ball_posession
                    FROM detected_objects
                    WHERE frame_id = ? AND class_name IN ("player", "goalkeeper", "ball")
                ''', (frame_id,))
                records = self.cursor.fetchall()

                # Convert records to dictionaries
                records = [
                    {
                        'frame_id': row[0],
                        'class_name': row[1],
                        'x_transformed_metres': row[2],
                        'y_transformed_metres': row[3],
                        'ball_posession': row[4]
                    }
                    for row in records
                ]

                # Calculate ball possession
                updated_records = self.calculate_ball_posession(records)

                # Perform updates
                for record in updated_records:
                    self.cursor.execute('''
                        UPDATE detected_objects
                        SET ball_posession = ?
                        WHERE frame_id = ? AND class_name = ? AND 
                              x_transformed_metres = ? AND y_transformed_metres = ?
                    ''', (
                        record['ball_posession'],
                        record['frame_id'],
                        record['class_name'],
                        record['x_transformed_metres'],
                        record['y_transformed_metres']
                    ))

                    processed_records += 1

                    # Commit changes in batches
                    if processed_records % self.batch_size == 0:
                        self.conn.commit()

                tracker_pbar.update(1)

        # Final commit
        self.conn.commit()
        self.conn.close()

         # TIME FOR FULL PROCESS: Calculate elapsed time for entire process:
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time
        minutes, seconds = divmod(int(total_time), 60)
        print(f"✅ Processing time: {minutes}:{seconds} sec")

'''
class BallPosessionAssigner():
    def __init__(self):
        self.max_player_ball_distance = 80

    # - 
    def is_in_posession_of_ball(self,players,ball_bbox):

        ball_position = self.get_center_of_bbox(ball_bbox)
        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in player.items():
            player_bbox = player['bbox']

            # get the distance between each foot of the player and the ball
            x = player_bbox[0]
            bottom_y = player_bbox[-1]
            x2 = player_bbox[2]
            distance_left_foot = self.measure_distance((x,bottom_y),ball_position)
            distance_right_foot = self.measure_distance((x2,bottom_y),ball_position)
            distance = min(distance_left_foot,distance_right_foot)

            # which player is close to the ball:
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
    
    def get_center_of_bbox(self,x1,y1,x2,y2):
        return int((x1+x2)/2),int((y1+y2)/2)

    def measure_distance(self,p1,p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
'''