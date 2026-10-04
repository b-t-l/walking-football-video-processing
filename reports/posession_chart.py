import matplotlib.pyplot as plt
from collections import defaultdict
import time
from utils import *

class PosessionChart:
    def __init__(self, possession_data, team_colors,possession_counts,ball_giveaways,successful_passes,CREATE_REPORT_OUTPUT_PATH,GAME_RECORD):
        self.possession_data = possession_data  
        self.team_colors = team_colors  # Dictionary {team: color}
        self.possession_counts = possession_counts
        self.ball_giveaways = ball_giveaways
        self.successful_passes = successful_passes
        self.CREATE_REPORT_OUTPUT_PATH = CREATE_REPORT_OUTPUT_PATH
        self.GAME_RECORD = GAME_RECORD



    ##########################################################################OVERALL POSESSION CHART:
    def generate_overall_posession_chart(self):

        # Teams and their possession data
        teams = list(self.possession_data.keys())
        possession_frames = list(self.possession_data.values())
        
        # Calculate total possession frames
        total_frames = sum(possession_frames)
        
        # Convert frames to percentages
        possession_percentages = [(frames / total_frames) * 100 for frames in possession_frames]

        # Get colors for the teams
        colors = [self.team_colors.get(team, (0.5, 0.5, 0.5)) for team in teams]  # Default to gray if missing

        # Create the bar chart
        fig, ax = plt.subplots()
        bars = plt.bar(teams, possession_percentages, color=colors)

        # Add percentages on top of each bar
        for bar, percentage in zip(bars, possession_percentages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Center of the bar
                height + 1,  # Slightly above the bar
                f'{percentage:.1f}%',  # Format percentage with 1 decimal
                ha='center',  # Align horizontally to center
                va='bottom',  # Align vertically to the top of the bar
                fontsize=10
            )

        # Update axis labels and title
        plt.xlabel('Team')
        plt.ylabel('Ball Possession (%)')
        plt.title('Ball Possession by Team')
        plt.ylim(0, 100)  # Y-axis limits for percentages
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        chart_path = f"{self.CREATE_REPORT_OUTPUT_PATH}/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-possession_chart-{timestr}.png"
        plt.savefig(chart_path)
        plt.close()

        return chart_path


    ##########################################################################OVERALL POSESSION CHART:
    def generate_posession_given_away_chart(self):

        '''
        work out how many times each team gives the ball away.
        - get all the records where ball_posession = true.
        - order them by frame_id
        - work out each time the team in posession is followed by the next posession NOT being the same team.
        - keep a running count fo these occurances by team
        - plot a bar chart with these totals.
        '''
        # Prepare data for the bar chart
        teams = list(self.ball_giveaways.keys())
        counts = list(self.ball_giveaways.values())
        colors = [self.team_colors.get(team, (0.5, 0.5, 0.5)) for team in teams]  # Default to gray if missing

        # Create the bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(teams, counts, color=colors)
        plt.title("Number of Ball Giveaways by Team", fontsize=16)
        plt.ylabel("Number of Giveaways", fontsize=12)
        plt.xlabel("Team", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=12)

        # Add total counts above each bar
        for bar, count in zip(bars, counts):
            yval = bar.get_height()  # Get the height of the bar
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{count}", ha='center', va='bottom', fontsize=10)

        # Save the chart
        timestr = time.strftime("%Y%m%d-%H%M%S")
        chart_path = f"{self.CREATE_REPORT_OUTPUT_PATH}/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-possession_given_away_chart-{timestr}.png"
        plt.savefig(chart_path)
        plt.close()

        return chart_path


    ##########################################################################OVERALL POSESSION CHART:
    def generate_successful_passes_chart(self):

        # Prepare data for the bar chart
        teams = list(self.successful_passes.keys())
        counts = list(self.successful_passes.values())
        colors = [self.team_colors.get(team, (0.5, 0.5, 0.5)) for team in teams]  # Default to gray if missing

        # Create the bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(teams, counts, color=colors)
        plt.title("Successful Passes by Team", fontsize=16)
        plt.ylabel("Number of passes", fontsize=12)
        plt.xlabel("Team", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=12)

        # Add total counts above each bar
        for bar, count in zip(bars, counts):
            yval = bar.get_height()  # Get the height of the bar
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{count}", ha='center', va='bottom', fontsize=10)

        # Save the chart
        timestr = time.strftime("%Y%m%d-%H%M%S")
        chart_path = f"{self.CREATE_REPORT_OUTPUT_PATH}/game-id-{self.GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(self.GAME_RECORD['title'])}-successful-passes-{timestr}.png"
        plt.savefig(chart_path)
        plt.close()

        return chart_path
    

    ##########################################################################
    def plot_possession_comparison(self, possession_counts, team_colors):
        """
        Plot a bar chart comparing ball possession between two teams in four quarters.

        Args:
            possession_counts (dict): A dictionary with possession counts per team for each quarter.
            teams (list): A list of two team names for comparison.
        """
        quarters = list(possession_counts.keys())
        team_1 = teams[0]
        team_2 = teams[1]

        # Extract possession counts for both teams
        team_1_possession = [possession_counts[q].get(team_1, 0) for q in quarters]
        team_2_possession = [possession_counts[q].get(team_2, 0) for q in quarters]

        # Create the bar chart
        x = np.arange(len(quarters))  # Quarter indices
        width = 0.35  # Width of each bar

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot bars for each team
        ax.bar(x - width / 2, team_1_possession, width, label=team_1, color='blue')
        ax.bar(x + width / 2, team_2_possession, width, label=team_2, color='red')

        # Add labels, title, and legend
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Ball Possession (Frames)', fontsize=12)
        ax.set_title('Ball Possession Comparison by Quarter', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Quarter {q}" for q in quarters])
        ax.legend()

        # Show values on top of each bar
        for idx, val in enumerate(team_1_possession):
            ax.text(idx - width / 2, val + 1, str(val), ha='center', va='bottom', fontsize=10)
        for idx, val in enumerate(team_2_possession):
            ax.text(idx + width / 2, val + 1, str(val), ha='center', va='bottom', fontsize=10)

        # Save and show the plot
        plt.tight_layout()
        plt.savefig(f"{self.CREATE_REPORT_OUTPUT_PATH}-possession_comparison_chart.png")
        plt.close()