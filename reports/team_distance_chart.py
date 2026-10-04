import matplotlib.pyplot as plt
import time
from utils import StringUtils  # Ensure you have this import if needed


class TeamDistanceChart:
    def __init__(self, distance_data, team_colors,CREATE_REPORT_OUTPUT_PATH,GAME_RECORD):
        
        self.distance_data = distance_data  # Dictionary {"Team A": value, "Team B": value}
        self.team_colors = team_colors      # Dictionary {"Team A": color, "Team B": color}
        self.CREATE_REPORT_OUTPUT_PATH = CREATE_REPORT_OUTPUT_PATH
        self.GAME_RECORD = GAME_RECORD

    def generate_chart(self):
        teams = list(self.distance_data.keys())
        distances = list(self.distance_data.values())

        # Calculate total distance
        total_distance = sum(distances)

        # Calculate percentages
        percentages = [(distance / total_distance) * 100 for distance in distances] if total_distance > 0 else [0] * len(distances)

        colors = [self.team_colors.get(team, (0.5, 0.5, 0.5)) for team in teams]  # Default to gray if missing

        plt.figure(figsize=(10, 6))
        bars = plt.bar(teams, percentages, color=colors)
        plt.title("Percentage of Total Distance Covered", fontsize=16)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add percentage values above each bar
        for bar, percentage in zip(bars, percentages):
            yval = bar.get_height()  # Get the height of the bar
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{percentage:.1f}%", ha='center', va='bottom', fontsize=10)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        chart_path = f"{self.CREATE_REPORT_OUTPUT_PATH}/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-team_distance_chart-{timestr}.png"
        plt.savefig(chart_path)
        plt.close()

        return chart_path
