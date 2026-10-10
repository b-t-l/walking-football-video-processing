import sqlite3
from tqdm import tqdm
from datetime import datetime
import time
from fpdf import FPDF
from collections import defaultdict
from reports.team_distance_chart import TeamDistanceChart
from reports.tracker_speed_chart import TrackerSpeedChart
from reports.tracker_speed_offenders_table import TrackerSpeedOffendersTable
from reports.posession_chart import PosessionChart
from utils import *


class CreateStatisticsReport:

    ###################################################################### INITIALIZE:
    def __init__(self, db_path, OUTPUT_PATH, TEAMS, GAME_RECORD, DETECTION_FPS):

        print("--------------------------------------------------------------------------------------------------")
        print("CREATE STATISTICS REPORT: generate stats report and save to pdf")
        print("--------------------------------------------------------------------------------------------------")

        self.db_path = db_path
        self.game_name = GAME_RECORD['title']
        self.game_date = GAME_RECORD['date']
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.CREATE_REPORT_OUTPUT_PATH = OUTPUT_PATH
        self.TEAMS = TEAMS
        self.TEAM_NAMES = list(self.TEAMS.keys())
        self.REPORT_CREATE_TIME = time.strftime("%Y%m%d-%H%M%S")
        self.GAME_RECORD = GAME_RECORD
        self.GAME_RUNNING_SPEED = float(GAME_RECORD['running_speed_km_h'])

        self.fps = DETECTION_FPS
        # Create the output video file name:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.REPORT_FILE = f"../output/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-stats-report-{timestr}.pdf"


    ###################################################################### FUNCTIONS:

    ########################### WHICH COLUMN IDENTIFIES ONE PERSON: player_id (stitched) if present, else tracker_id
    def person_id_column(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(detected_objects)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        return "player_id" if "player_id" in columns else "tracker_id"

    ########################### FETCH DATA FROM DATABASE and create datasets used in each chart:
    def fetch_data(self):
        print("Fetching data from the database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # query to get the db games data:

        # Query to fetch player data
        query = f'''
        SELECT {self.person_id_column()}, team, team_color, total_distance_metres, speed_km_per_hour, ball_posession, frame_id, x_transformed_metres, y_transformed_metres
        FROM detected_objects
        WHERE team IN ({', '.join('?' for _ in self.TEAM_NAMES)}) 
        AND team NOT IN ("Referee", "Goalkeeper") 
        AND is_inside_pitch = 1 
        AND class_name != "ball"
        '''
        cursor.execute(query,self.TEAM_NAMES)
        results = cursor.fetchall()
        conn.close()

        # Process data into a useful structure
        distance_data = {}
        max_speeds = {}
        tracker_colors = {}
        possession_data = {}
        team_colors = {}
        speed_details = []

        # Initialize a dictionary to count ball giveaways
        ball_giveaways = defaultdict(int)
        successful_passes = defaultdict(int)
        previous_team = None

        # Calculate possession
        field_size = (46, 21)  # Field dimensions in meters
        possession_counts = self.calculate_possession_quarters(results, field_size)

        # Print possession counts per quarter
        for quarter, counts in possession_counts.items():
            print(f"Quarter {quarter}:")
            for team, count in counts.items():
                print(f"  Team {team}: {count} frames of possession")

        ## FROM THE RESULTS CALCULATE EACH DATASET WE WILL NEED TO CREATE CHARTS FROM:
        for row in tqdm(results, desc="Processing player data", unit="record"):
            tracker_id, team, team_color, distance, speed, ball_posession, frame_id, x_transformed_metres, y_transformed_metres = row

            # DISTANCE: calculated separately (see calculate_team_distances) - the database column
            # total_distance_metres is a RUNNING total per tracker, so it must not be summed row by row.

            # SPEED: Track max speed per tracker ID
            if speed is not None:  # Check if speed is not None
                if tracker_id not in max_speeds or speed > max_speeds[tracker_id]:
                    max_speeds[tracker_id] = float(speed)  # Ensure speed is stored as a float

            # SPEED: Collect speed details for the chart
            if speed:  # Ensure speed is not None or 0
                speed_details.append((frame_id, team, tracker_id, speed))

            # TEAMS: Convert team_color to normalized RGB and use tracker_id as the key
            if team_color:
                cleaned_color = team_color.strip("()")
                bgr_values = [int(x) for x in cleaned_color.split(",")]
                rgb_values = self.bgr_to_rgb(tuple(bgr_values))  # Convert BGR to RGB
                tracker_colors[tracker_id] = tuple(c / 255 for c in rgb_values)  # Normalize to 0-1 range
            else:
                tracker_colors[tracker_id] = (0.5, 0.5, 0.5)  # Default gray color if missing

            if team_color:
                cleaned_color = team_color.strip("()")
                bgr_values = [int(x) for x in cleaned_color.split(",")]
                rgb_values = self.bgr_to_rgb(tuple(bgr_values))  # Convert BGR to RGB
                team_colors[team] = tuple(c / 255 for c in rgb_values)  # Normalize to 0-1 range
            else:
                team_colors[team] = (0.5, 0.5, 0.5)  # Default gray color if missing

            # BALL POSESSION: Track overall ball possession data
            if ball_posession:
                if team not in possession_data:
                    possession_data[team] = 0
                possession_data[team] += 1  # Increment possession frames for the team

            # BALL POSESSION: Track give away the ball ( includes passing the into sideline)
            if ball_posession:
                # we only take it as a give away if the change happend within a teim frame ( so we can exlucde, goals, free kicks etc)
                DURATION_THRESHOLD_IN_FRAMES = self.fps * 5
                if previous_team and previous_team != team:
                    # Check if the time difference is within the specified duration
                    if frame_id - last_possession_change_frame <= DURATION_THRESHOLD_IN_FRAMES:
                        ball_giveaways[previous_team] += 1  # Increment giveaway count for the previous team
                # Update the last possession change frame
                last_possession_change_frame = frame_id
                previous_team = team

            # BALL POSESSION: successful passes to players on the same team:
            if ball_posession:
                # the ball has to reach the player in under 3 seconds
                DURATION_THRESHOLD_IN_FRAMES = self.fps * 3
                if previous_team == team:  # Check if the current team is the same as the previous team
                    # Check if the time difference is within the specified duration
                    if frame_id - last_possession_change_frame <= DURATION_THRESHOLD_IN_FRAMES:
                        successful_passes[team] += 1  # Increment successful passes for the current team
                # Update the last possession change frame
                last_possession_change_frame = frame_id
                previous_team = team

        distance_data = self.calculate_team_distances()

        return distance_data, max_speeds, tracker_colors, possession_data, team_colors, speed_details, possession_counts, ball_giveaways, successful_passes


    ########################### DISTANCE PER TEAM:
    def calculate_team_distances(self):
        """
        total_distance_metres is a RUNNING total for each tracker_id (it grows frame by frame), so the
        distance covered in one frame is the change from that tracker's previous row. Summing those
        per-frame changes by the team shown on each row gives the real distance covered per team,
        and still works when the tracker's team label flips part-way through a track.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        id_column = self.person_id_column()
        cursor.execute(f'''
            SELECT {id_column}, team, total_distance_metres
            FROM detected_objects
            WHERE class_name = "player"
            ORDER BY {id_column}, frame_id, _id
        ''')
        rows = cursor.fetchall()
        conn.close()

        distance_data = {}
        previous_tracker = None
        previous_total = 0.0
        for tracker_id, team, total in rows:
            total = total or 0.0
            if tracker_id != previous_tracker:
                previous_total = 0.0
                previous_tracker = tracker_id
            step = max(total - previous_total, 0.0)
            previous_total = total
            if team in self.TEAM_NAMES and team not in ("Referee", "Goalkeeper"):
                distance_data[team] = distance_data.get(team, 0.0) + step
        return distance_data


    ########################### ADD PAGES TO THE PDF:
    def add_title_page(self):

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=24, style="B")
        self.pdf.cell(0, 20, "Game Statistics", ln=True, align="C")

        self.pdf.set_font("Arial", size=14, style="I")
        self.pdf.cell(0, 10, f"{self.GAME_RECORD['game_format']} - {self.GAME_RECORD['game_type']}", ln=True, align="C")

        self.pdf.set_font("Arial", size=16, style="B")
        self.pdf.cell(0, 20, self.GAME_RECORD['title'], ln=True, align="C")

        self.pdf.set_font("Arial", size=20, style="B")
        self.pdf.cell(0, 10, f"Score: {self.GAME_RECORD['score']}", ln=True, align="C")

        self.pdf.set_font("Arial", size=12, style="I")
        self.pdf.cell(0, 10, f"Game Date: {self.GAME_RECORD['date']}", ln=True, align="C")

        self.pdf.set_font("Arial", size=12, style="I")
        self.pdf.cell(0, 10, f"Report created: {self.REPORT_CREATE_TIME}", ln=True, align="C")

        self.pdf.set_font("Arial", size=12, style="I")
        self.pdf.cell(0, 10,'Click here to analysis video', link=f"{self.GAME_RECORD['statistics_output_video']}", ln=True, align="C")
        
        self.pdf.set_font("Arial", size=12, style="I")
        self.pdf.cell(0, 10, f"Video chapters:", ln=True, align="C")

        video_keypoints = self.calculate_video_keypoints()

    #### OVEREALL VIDEO KEYPOINTS: start,halftime,finish,goals etc
    def calculate_video_keypoints(self):
        print(self.GAME_RECORD)
        text = f""

        source_fps = 24  # Source video frame rate
        detection_fps = 6  # Created video frame rate

        # Extract times from the game data and convert to seconds
        game_start_seconds = TimeUtils.time_to_seconds_from_game_data(self.GAME_RECORD['game_start_seconds'])
        half_time_start_seconds = TimeUtils.time_to_seconds_from_game_data(self.GAME_RECORD['half_time_start_seconds'])
        half_time_end_seconds = TimeUtils.time_to_seconds_from_game_data(self.GAME_RECORD['half_time_end_seconds'])
        game_end_seconds = TimeUtils.time_to_seconds_from_game_data(self.GAME_RECORD['game_end_seconds'])

        # Adjust times based on the new video consider time periods to exclude:
        adjusted_game_start = game_start_seconds - game_start_seconds
        adjusted_half_time_start = half_time_start_seconds - game_start_seconds
        adjusted_game_end = game_end_seconds - game_start_seconds

        # Convert back to mm:ss format
        adjusted_game_start_mm_ss = TimeUtils.seconds_to_mm_ss(adjusted_game_start)
        adjusted_half_time_mm_ss = TimeUtils.seconds_to_mm_ss(adjusted_half_time_start)
        #half_time_end_mm_ss = TimeUtils.seconds_to_mm_ss(adjusted_half_time_end)
        adjusted_game_end_mm_ss = TimeUtils.seconds_to_mm_ss(adjusted_game_end)

        # Print results
        print(f"game_start_seconds: {game_start_seconds}")
        print(f"half_time_start_seconds: {half_time_start_seconds}")
        print(f"game_end_seconds: {game_end_seconds}")
        
        print(f"adjusted_game_start_mm_ss: {adjusted_game_start_mm_ss}")
        print(f"adjusted_half_time_mm_ss: {adjusted_half_time_mm_ss}")
        #print(f"Half Time End: {half_time_end_mm_ss}")
        print(f"adjusted_game_end_mm_ss: {adjusted_game_end_mm_ss}")

        # create the youtube description:
        self.pdf.cell(0, 10,'Start', link=f"{self.GAME_RECORD['statistics_output_video']}&t={adjusted_game_start}s", ln=True, align="C")
        self.pdf.cell(0, 10,'Half time', link=f"{self.GAME_RECORD['statistics_output_video']}&t={adjusted_half_time_start}s", ln=True, align="C")
        self.pdf.cell(0, 10,'End game', link=f"{self.GAME_RECORD['statistics_output_video']}&t={adjusted_game_end}s", ln=True, align="C")


    def check_sustained_speed(self,speed_details):
        # Group speed details by player (team + tracker_id)
        player_speeds = {}
        for frame_id, team, tracker_id, speed in speed_details:
            player_key = (team, tracker_id)
            if player_key not in player_speeds:
                player_speeds[player_key] = []
            player_speeds[player_key].append((frame_id, speed))

        # Check for sustained high speeds
        sustained_speeds = []
        for (team, tracker_id), speeds in player_speeds.items():
            speeds.sort()  # Sort by frame_id
            consecutive_count = 0
            start_frame = None
            
            for frame_id, speed in speeds:
                if speed > 15:
                    if consecutive_count == 0:
                        start_frame = frame_id
                    consecutive_count += 1
                    
                    # we only count it as running if the speed is above threshold for 1 second:
                    if consecutive_count >= self.fps/2:
                        # Convert frame_id to time (mm:ss)
                        time_seconds = start_frame / 24
                        time_mm_ss = f"{int(time_seconds // 60):02}:{int(time_seconds % 60):02}"
                        sustained_speeds.append(
                            f"{time_mm_ss} -- {team} player {tracker_id} sustained speed above 15 km/h for {self.fps/2} frames (speed: {speed:.2f} km/h)"
                        )
                        consecutive_count = 0  # Reset to avoid duplicate entries
                else:
                    consecutive_count = 0
                    start_frame = None

        return sustained_speeds

    ### YOUTUBE: description file:
    def write_youtube_description_text_file(self, max_speeds, team_colors, tracker_colors, speed_details):

        timestr = time.strftime("%Y%m%d-%H%M%S")
        text_file_name = f"../output/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-youtube-description-{timestr}.txt"
        
        ### create the start /half / end
        text_lines = [
            f"GAME DETAILS  -----------------",
            f" ",
            f"{self.GAME_RECORD['game_start_seconds']} -- Start game",
            f"{self.GAME_RECORD['half_time_start_seconds']} -- Half time",
            f"{self.GAME_RECORD['half_time_end_seconds']} -- 2nd Half start ",
            f"{self.GAME_RECORD['game_end_seconds']} -- End game"
        ]

        ### add the running chapters for the running (timestamp)
        text_lines.append(f" ")
        text_lines.append(f"SPEED DETECTIONS ------------------------")
        text_lines.append(f" ")

        # Append each time speed of player is above 15 km/h in the format mm:ss
        text_lines.extend(self.check_sustained_speed(speed_details))

        ### add space to start adding the Player pointers:
        text_lines.append(f" ")
        text_lines.append(f"TEAM ANALYSIS ------------------------")
        text_lines.append(f" ")

        ### add space to start adding the Player pointers:
        text_lines.append(f" ")
        text_lines.append(f"PLAYER ANALYSIS ------------------------")
        text_lines.append(f" ")
        # write the file
        with open(text_file_name, 'w') as file:
            for line in text_lines:
                file.write(line + '\n')  # Write each line followed by a newline character


    ### CHARTS: add all the posession charts: 
    def add_possession_chart_page(self, possession_data,team_colors,possession_counts,ball_giveaways,successful_passes):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 10, "-- POSESSION --", ln=True, align="C")
        # Generate the possession charts
        chart = PosessionChart(possession_data, team_colors, possession_counts,ball_giveaways,successful_passes,self.CREATE_REPORT_OUTPUT_PATH,self.GAME_RECORD)
        chart_path = chart.generate_overall_posession_chart()
        self.pdf.image(chart_path, x=10, y=40, w=190)

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 10, "-- POSESSION --", ln=True, align="C")
        chart_path = chart.generate_posession_given_away_chart()
        self.pdf.image(chart_path, x=10, y=40, w=190)

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 10, "-- POSESSION --", ln=True, align="C")
        chart_path = chart.generate_successful_passes_chart()
        self.pdf.image(chart_path, x=10, y=40, w=190)
        

    ### CHARTS: add all the distance charts: 
    def add_distance_chart_page(self, distance_data, team_colors):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 10, "-- DISTANCE --", ln=True, align="C")

        chart = TeamDistanceChart(distance_data, team_colors,self.CREATE_REPORT_OUTPUT_PATH,self.GAME_RECORD)
        chart_path = chart.generate_chart()
        self.pdf.image(chart_path, x=10, y=40, w=190)

    ### CHARTS: add all the speed charts: 
    def add_speed_chart_page(self, speed_data, team_colors, tracker_colors, speed_details, fps):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        
        # Ensure that tracker_colors is being used with tracker_id
        self.pdf.cell(0, 10, "-- SPEED --", ln=True, align="C")
        tracker_chart = TrackerSpeedChart(speed_data, team_colors, tracker_colors, speed_details, fps,self.CREATE_REPORT_OUTPUT_PATH,self.GAME_RECORD)
        # Generate the speed count chart
        speed_count_chart_path = tracker_chart.generate_speed_count_chart()
        self.pdf.image(speed_count_chart_path, x=10, y=40, w=190)
        
        # Generate the max speed chart
        #self.pdf.add_page()
        #self.pdf.cell(0, 10, "-- SPEED --", ln=True, align="C")
        #max_speed_chart_path = tracker_chart.generate_speed_offenders_chart()
        #self.pdf.image(max_speed_chart_path, x=10, y=40, w=190)
        # Now you can include max_speed_chart_path and speed_count_chart_path in your PDF report

    def add_speed_offenders_table(self, speed_data, team_colors, tracker_colors, speed_details, fps):
        print('add_speed_offenders_table')
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        # Ensure that tracker_colors is being used with tracker_id
        self.pdf.cell(0, 10, "-- SPEED --", ln=True, align="C")
        # Generate the speed offenders table:
        tracker_speed_table = TrackerSpeedOffendersTable(speed_data, team_colors, tracker_colors, speed_details, fps,self.CREATE_REPORT_OUTPUT_PATH,self.GAME_RECORD)
        # add the table here: this table shows any players who have a speed above 15 km/h

    ### CHARTS: create the pdf report
    def save_pdf(self):
        self.pdf.output(self.REPORT_FILE)


    #### USEFUL FUNCTIONS: Convert colors:
    def bgr_to_rgb(self,bgr_color):
        """
        Convert a BGR color (used in OpenCV) to an RGB color (used in FPDF).
        
        Args:
            bgr_color (tuple): A tuple representing the BGR color (B, G, R).

        Returns:
            tuple: A tuple representing the RGB color (R, G, B).
        """
        return (bgr_color[2], bgr_color[1], bgr_color[0])  # Swap B and R


    ########################### Posession across quarters of the field
    def calculate_possession_quarters(self, results, field_size=(21, 46)):
        """
        Calculate ball possession for each team in the four field quarters.

        Args:
            results (list): List of tuples with query results.
            field_size (tuple): Dimensions of the field (width, height) in meters.

        Returns:
            dict: A dictionary with possession counts per team for each quarter.
        """
        field_width, field_height = field_size

        # Define field quarters
        quarters = {
            1: lambda x, y: x <= field_width / 2 and y <= field_height / 2,  # Top Left
            2: lambda x, y: x > field_width / 2 and y <= field_height / 2,   # Top Right
            3: lambda x, y: x <= field_width / 2 and y > field_height / 2,  # Bottom Left
            4: lambda x, y: x > field_width / 2 and y > field_height / 2,   # Bottom Right
        }

        # Initialize possession counts
        possession_counts = {
            1: {},  # Quarter 1
            2: {},  # Quarter 2
            3: {},  # Quarter 3
            4: {},  # Quarter 4
        }

        # Process results
        for row in results:
            tracker_id, team, _, _, _, ball_possession, _, x_transformed, y_transformed = row

            # Only process rows with ball possession
            if not ball_possession:
                continue

            # Determine the quarter
            for quarter, condition in quarters.items():
                if condition(x_transformed, y_transformed):
                    # Initialize team count if not present
                    if team not in possession_counts[quarter]:
                        possession_counts[quarter][team] = 0
                    # Increment possession count
                    possession_counts[quarter][team] += 1
                    break

        return possession_counts    


    ###################################################################### RUN:
    def run(self):
        
        # Fetch data from the database
        distance_data, max_speeds, tracker_colors, possession_data, team_colors, speed_details, possession_counts, ball_giveaways, successful_passes = self.fetch_data()

        # Write the yourtube chapters file as text:
        self.write_youtube_description_text_file(max_speeds, team_colors, tracker_colors, speed_details)

        # Add pages to pdf:
        self.add_title_page()
        self.add_possession_chart_page(possession_data,team_colors,possession_counts,ball_giveaways,successful_passes)  # Add possession chart
        self.add_distance_chart_page(distance_data,team_colors)
        self.add_speed_chart_page(max_speeds, team_colors, tracker_colors, speed_details, self.fps)
        self.add_speed_offenders_table(max_speeds, team_colors, tracker_colors, speed_details, self.fps)
        # Save PDF
        self.save_pdf()
        
        print(f"Report generated: {self.REPORT_FILE}")
        print("-------------------------------------------------")