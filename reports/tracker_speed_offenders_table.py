import matplotlib.pyplot as plt
import numpy as np
from utils import *
import time

class TrackerSpeedOffendersTable:
    def __init__(self, speed_data, team_colors, tracker_colors, speed_details, fps,CREATE_REPORT_OUTPUT_PATH,GAME_RECORD):
        """
        Args:
            speed_data: Dictionary {tracker_id: speed}
            tracker_colors: Dictionary {tracker_id: color}
            speed_details: List of tuples [(frame_id, team_name, tracker_id, speed)]
            fps: Frames per second of the video
        """
        self.speed_data = speed_data  # Dictionary {tracker_id: speed}
        self.tracker_colors = tracker_colors  # Dictionary {tracker_id: color}
        self.team_colors = team_colors  # Dictionary {tracker_id: color}
        self.speed_details = speed_details  # List of details for all tracked speeds
        self.fps = fps  # Frames per second of the video
        self.CREATE_REPORT_OUTPUT_PATH = CREATE_REPORT_OUTPUT_PATH
        self.GAME_RECORD = GAME_RECORD

    def generate_speed_offenders_table(self):

        print('...generate_speed_offenders_table(self)')
        
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 10, "-- SPEED OFFENDERS --", ln=True, align="C")

        # Filter and sort speed details for players with speed above 15 km/h
        offenders = [detail for detail in self.speed_details if detail[3] > 15]
        offenders.sort(key=lambda x: x[3], reverse=True)  # Sort by speed in descending order

        # Add table headers
        self.pdf.set_font("Arial", size=10, style="B")
        self.pdf.cell(30, 10, "Frame ID", 1)
        self.pdf.cell(30, 10, "Time (mm:ss)", 1)
        self.pdf.cell(30, 10, "Team", 1)
        self.pdf.cell(30, 10, "Tracker ID", 1)
        self.pdf.cell(30, 10, "Speed (km/h)", 1)
        self.pdf.ln()

        # Add table rows
        self.pdf.set_font("Arial", size=10)
        for frame_id, team, tracker_id, speed in offenders:
            # Convert frame_id to time (mm:ss)
            time_seconds = frame_id / self.fps
            time_mm_ss = f"{int(time_seconds // 60):02}:{int(time_seconds % 60):02}"

            self.pdf.cell(30, 10, str(frame_id), 1)
            self.pdf.cell(30, 10, time_mm_ss, 1)
            self.pdf.cell(30, 10, team, 1)
            self.pdf.cell(30, 10, str(tracker_id), 1)
            self.pdf.cell(30, 10, f"{speed:.2f}", 1)
            self.pdf.ln()
