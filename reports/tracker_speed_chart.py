import matplotlib.pyplot as plt
import numpy as np
from utils import *
import time

class TrackerSpeedChart:
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

    def generate_max_speed_chart(self):
        # Extract tracker IDs and speeds for the bar chart
        tracker_ids = list(self.speed_data.keys())
        max_speeds = list(self.speed_data.values())
        colors = [self.tracker_colors[tracker] for tracker in tracker_ids]

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create the bar chart for max speeds
        ax.bar(tracker_ids, max_speeds, color=colors)
        ax.set_title("Max Speed by Player ID (km/h)", fontsize=16)
        ax.set_ylabel("Speed (km/h)", fontsize=12)
        ax.set_xlabel("Player ID", fontsize=12)
        ax.set_xticks(np.arange(len(tracker_ids)))
        ax.set_xticklabels(tracker_ids, fontsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=12)

        # Add horizontal lines for records
        ax.axhline(y=20.2, color='red', linestyle='--', linewidth=1.5)
        ax.text(len(tracker_ids) - 0.5, 20.4, 'Mens World Record', color='red', fontsize=9, ha='right', va='top')
        ax.axhline(y=17.5, color='blue', linestyle='--', linewidth=1.5)
        ax.text(len(tracker_ids) - 0.5, 17.5, 'Mens British Record', color='blue', fontsize=9, ha='right', va='top')

        # Save the max speed chart
        timestr = time.strftime("%Y%m%d-%H%M%S")
        max_speed_chart_path = f"{self.CREATE_REPORT_OUTPUT_PATH}/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-max_speed_chart-{timestr}.png"
        plt.savefig(max_speed_chart_path)
        plt.close()

        return max_speed_chart_path

    def generate_speed_count_chart(self):
        # Calculate the number of times each player's speed was above 15 km/h
        team_speed_counts = {}
        for frame_id, team_name, tracker_id, speed in self.speed_details:
            if speed > 15:
                if team_name not in team_speed_counts:
                    team_speed_counts[team_name] = 0
                team_speed_counts[team_name] += 1

        # Prepare data for the new bar chart
        teams = list(team_speed_counts.keys())
        counts = [team_speed_counts.get(team, 0) for team in teams]
        #team_colors = [self.tracker_colors.get(team, (0.5, 0.5, 0.5)) for team in teams]  # Get colors for the teams
        team_colors = [self.team_colors.get(team, (0.5, 0.5, 0.5)) for team in teams]  # Default to gray if missing
        # Create a new figure and axis for the count chart
        fig, ax2 = plt.subplots(figsize=(10, 6))  # Adjust size as needed

        # Create the bar chart for counts
        bars = ax2.bar(teams, counts, color=team_colors, alpha=1)
        ax2.set_title("Count of Speeds Above 15 km/h", fontsize=16)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_xticks(np.arange(len(teams)))
        ax2.set_xticklabels(teams, fontsize=10, rotation=45)
        ax2.tick_params(axis='y', labelsize=12)

        # Add total counts above each bar in the count chart
        for bar, count in zip(bars, counts):
            yval = bar.get_height()  # Get the height of the bar
            ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{count}", ha='center', va='bottom', fontsize=10)

        # Save the speed count chart
        timestr = time.strftime("%Y%m%d-%H%M%S")
        speed_count_chart_path = f"{self.CREATE_REPORT_OUTPUT_PATH}/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-speed_count_chart-{timestr}.png"
        plt.savefig(speed_count_chart_path)
        plt.close()

        return speed_count_chart_path

    def generate_speed_offenders_chart(self):
        # Filter speed data to include only speeds above 10 km/h
        filtered_speed_data = {tracker_id: speed for tracker_id, speed in self.speed_data.items() if speed > 10}

        # Extract tracker IDs and speeds for the bar chart
        tracker_ids = list(filtered_speed_data.keys())
        max_speeds = list(filtered_speed_data.values())
        colors = [self.tracker_colors[tracker] for tracker in tracker_ids if tracker in self.tracker_colors]  # Ensure colors match

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))  # Adjust height to accommodate the table

        # Create the bar chart for max speeds
        if tracker_ids:  # Check if there are any tracker IDs to plot
            ax.bar(tracker_ids, max_speeds, color=colors)
            ax.set_title("Max Speed by Player ID (km/h)", fontsize=16)
            ax.set_ylabel("Speed (km/h)", fontsize=12)
            ax.set_xlabel("Player ID", fontsize=12)
            ax.set_xticks(np.arange(len(tracker_ids)))
            ax.set_xticklabels(tracker_ids, fontsize=10, rotation=45)
            ax.tick_params(axis='y', labelsize=12)

            # Add horizontal lines for records
            ax.axhline(y=20.2, color='red', linestyle='--', linewidth=1.5)
            ax.text(len(tracker_ids) - 0.5, 20.4, 'Mens World Record', color='red', fontsize=9, ha='right', va='top')
            ax.axhline(y=17.5, color='blue', linestyle='--', linewidth=1.5)
            ax.text(len(tracker_ids) - 0.5, 17.5, 'Mens British Record', color='blue', fontsize=9, ha='right', va='top')
        else:
            ax.text(0.5, 0.5, 'No players with speed above 10 km/h', ha='center', va='center', fontsize=12)

        # Calculate the number of times each player's speed was above 15 km/h
        team_speed_counts = {}
        for frame_id, team_name, tracker_id, speed in self.speed_details:
            if speed > 15:
                if team_name not in team_speed_counts:
                    team_speed_counts[team_name] = 0
                team_speed_counts[team_name] += 1

        # Adjust layout to make space for the table
        plt.subplots_adjust(left=0.1, bottom=0.4, top=0.85)  # Adjust top to make space for the new chart

        # Filter speed details for table (speed > 20.2 km/h)
        filtered_details = [
            (frame_id, divmod(int(frame_id / self.fps), 60), team_name, tracker_id, speed)
            for frame_id, team_name, tracker_id, speed in self.speed_details
            if speed > 20.2
        ]

        # Prepare table data
        if filtered_details:
            table_data = [
                ["Frame ID", "Time (mm:ss)", "Team", "Tracker ID", "Speed (km/h)"]
            ] + [
                [
                    frame_id,
                    f"{time[0]:02}:{time[1]:02}",  # Format time as mm:ss
                    team_name,
                    tracker_id,
                    f"{speed:.1f}"
                ]
                for frame_id, time, team_name, tracker_id, speed in filtered_details
            ]
        else:
            table_data = [["No data available above 20.2 km/h"]]

        # Add table
        ax_table = plt.gca().table(
            cellText=table_data,
            colWidths=[0.2] * len(table_data[0]),
            cellLoc='center',
            loc='bottom',
            bbox=[0.0, -0.6, 1, 0.5]  # Adjust position and size of the table
        )
        ax_table.auto_set_font_size(False)
        ax_table.set_fontsize(8)  # Smaller font size for the table

        # Add padding to cells
        for key, cell in ax_table.get_celld().items():
            cell.set_edgecolor('lightgray')  # Light gray borders for cells
            cell.set_linewidth(0.5)
            cell.PAD = 10  # Add padding

        # Save the chart with the table
        timestr = time.strftime("%Y%m%d-%H%M%S")
        chart_path = f"{self.CREATE_REPORT_OUTPUT_PATH}/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-speed-offenders-{timestr}.png"
        plt.savefig(chart_path)
        plt.close()

        return chart_path
