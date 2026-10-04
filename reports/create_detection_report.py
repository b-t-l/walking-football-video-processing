import sqlite3
from tqdm import tqdm
from datetime import datetime
import time
from fpdf import FPDF
from utils import *

class CreateDetectionsReport:

    ########## INITIALIZE THE REPORT CREATOR:
    def __init__(self, 
                 db_path,
                 OUTPUT_PATH, 
                 TEAMS, 
                 GAME_RECORD, 
                 DETECTION_FPS):

        print("-------------------------------------------------")
        print("CREATE DETECTION REPORT: generate detection report and save to pdf")

        self.db_path = db_path
        self.game_name = GAME_RECORD['title']
        self.game_date = GAME_RECORD['date']
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.CREATE_REPORT_OUTPUT_PATH = OUTPUT_PATH
        self.TEAMS = TEAMS
        self.REPORT_CREATE_TIME = time.strftime("%Y%m%d-%H%M%S")
        self.GAME_RECORD = GAME_RECORD

        print(self.TEAMS)
        # get the games details from the database:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
                    SELECT _id, game_id, video, video_fps, detection_fps
                    FROM game
                    WHERE game_id = ?
                ''', (self.GAME_RECORD['game_id'],))
        self.DB_GAME_DETAILS = cursor.fetchone()
        self.DETECTION_FPS = self.DB_GAME_DETAILS[4]
        self.fps = self.DETECTION_FPS


    ########## RUN THE REPORT CREATOR:
    def run(self):
        
        start_time = time.time()  # Start the timer for overall process time
        # Fetch data from the database
        ball_count, goalkeeper_valid_frames, player_valid_frames, team_valid_frames, total_frames = self.fetch_data()
        
        # Add pages
        self.add_title_page(ball_count, goalkeeper_valid_frames, player_valid_frames, team_valid_frames, total_frames)

        # Save PDF
        self.save_pdf(f"game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-detection-report-{self.REPORT_CREATE_TIME}.pdf")

        # TIME FOR FULL PROCESS: Calculate elapsed time for entire process:
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time
        minutes, seconds = divmod(int(total_time), 60)
        print(f"✅ Processing time: {minutes}:{seconds} sec")
        print(f"Report generated: output/{self.game_name}-detection-report-{self.REPORT_CREATE_TIME}.pdf")
        print("-------------------------------------------------")


    ########## FETCH DATA FROM DB:
    def fetch_data(self):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query to fetch the class_name counts per unique frame_id
        query = '''
        SELECT frame_id, class_name, team
        FROM detected_objects
        WHERE is_inside_pitch = 1
        '''
        cursor.execute(query)
        results = cursor.fetchall()

        conn.close()

        # Dictionary to store counts of each class per frame_id
        frame_class_counts = {}

        # Group results by frame_id
        for frame_id, class_name, team in results:
            if frame_id not in frame_class_counts:
                frame_class_counts[frame_id] = {
                    'ball': 0, 
                    'goalkeeper': 0, 
                    'player': 0,
                    # Initialize counters for each team
                    **{team_name: 0 for team_name in self.TEAMS}
                }
            
            if class_name == 'ball':
                frame_class_counts[frame_id]['ball'] += 1
            elif team == 'Goalkeeper':
                frame_class_counts[frame_id]['goalkeeper'] += 1
            elif class_name == 'player'and team in self.TEAMS:
                frame_class_counts[frame_id]['player'] += 1
                frame_class_counts[frame_id][team] += 1  # Count per team

        # Now calculate statistics based on frame_class_counts
        ball_count = 0
        goalkeeper_valid_frames = 0
        player_valid_frames = 0
        team_valid_frames = {team_name: 0 for team_name in self.TEAMS}  # Track valid frames per team
        total_frames = len(frame_class_counts)

        for counts in frame_class_counts.values():
            ball_count += counts['ball']
            # Goalkeeper valid frame: check if there are exactly 2 goalkeepers in a frame
            if counts['goalkeeper'] == 2:
                goalkeeper_valid_frames += 1
            # Player valid frame: check if there are exactly 10 players in a frame (5 per team)
            if counts['player'] == 10:
                player_valid_frames += 1
            # Check for 5 players per team
            for team_name in self.TEAMS:
                if counts[team_name] == 5:
                    team_valid_frames[team_name] += 1

        # Return the counts and total number of frames
        return ball_count, goalkeeper_valid_frames, player_valid_frames, team_valid_frames, total_frames



    def add_title_page(self,ball_count, goalkeeper_valid_frames, player_valid_frames, team_valid_frames, total_frames):

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=24, style="B")
        self.pdf.cell(0, 20, "Game Detection Report", ln=True, align="C")

        self.pdf.set_font("Arial", size=14, style="I")
        self.pdf.cell(0, 10, f"{self.GAME_RECORD['game_format']} - {self.GAME_RECORD['game_type']}", ln=True, align="C")

        self.pdf.set_font("Arial", size=16, style="I")
        self.pdf.cell(0, 20, self.GAME_RECORD['title'], ln=True, align="C")

        self.pdf.set_font("Arial", size=12, style="I")
        self.pdf.cell(0, 10, f"Game Date: {self.GAME_RECORD['date']}", ln=True, align="C")

        self.pdf.set_font("Arial", size=12, style="I")
        self.pdf.cell(0, 10, f"Report created: {self.REPORT_CREATE_TIME}", ln=True, align="C")

        self.pdf.set_font("Arial", size=12, style="I")
        self.pdf.cell(0, 10, f"Video: {self.GAME_RECORD['youtube_360_video']}", ln=True, align="C")

        self.pdf.set_font("Arial", size=12, style="I")
        self.pdf.cell(0, 10, f"Detection fps: {self.DETECTION_FPS}", ln=True, align="C")

        # detection stats here:
        # 1. Ball detections percentage
        ball_percentage = (ball_count / total_frames) * 100
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 10, f"Ball detection frequency: {ball_count}/{total_frames} frames ({ball_percentage:.2f}%)", ln=True)

        # 2. Goalkeeper detections percentage
        goalkeeper_percentage = (goalkeeper_valid_frames / total_frames) * 100
        self.pdf.cell(0, 10, f"Goalkeeper detection frequency (2 per frame): {goalkeeper_valid_frames}/{total_frames} frames ({goalkeeper_percentage:.2f}%)", ln=True)

        # 3. Player detections percentage
        player_percentage = (player_valid_frames / total_frames) * 100
        self.pdf.cell(0, 10, f"Player detection frequency (10 players per frame): {player_valid_frames}/{total_frames} frames ({player_percentage:.2f}%)", ln=True)

        for team_name, valid_frames in team_valid_frames.items():
            team_percentage = (valid_frames / total_frames) * 100
            self.pdf.cell(0, 10, f"{team_name} (5 players): {valid_frames}/{total_frames} frames ({team_percentage:.2f}%)", ln=True)

    def save_pdf(self, filename):
        self.pdf.output(self.CREATE_REPORT_OUTPUT_PATH +'/'+ filename)


    def bgr_to_rgb(self,bgr_color):
        return (bgr_color[2], bgr_color[1], bgr_color[0])  # Swap B and R
