import pandas as pd

class TimeUtils:
    @staticmethod
    def time_to_seconds(time_string):
        """Convert a time string (MM:SS) to total seconds."""
        minutes, seconds = map(int, time_string.split(':'))
        return minutes * 60 + seconds

    @staticmethod
    def seconds_to_time(seconds):
        """Convert total seconds into a MM:SS formatted string."""
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"
    
    # Function to convert time string to total seconds
    def time_to_seconds_from_game_data(time_str):
        if pd.isna(time_str) or not str(time_str).strip():
            return None
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 2:  # MM:SS format
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:  # HH:MM:SS format
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return None

    # Function to convert total seconds to mm:ss format
    @staticmethod
    def seconds_to_mm_ss(total_seconds):
        if total_seconds is None:
            return "N/A"
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02}:{seconds:02}"

    # Function to adjust time based on frame ratio
    @staticmethod
    def adjust_time_for_new_video(original_time_seconds,source_fps,detection_fps):

        # Calculate the frame ratio
        frame_ratio = source_fps / detection_fps

        if original_time_seconds is None:
            return None
        # Convert original time to new video time
        adjusted_time_seconds = original_time_seconds / frame_ratio
        return adjusted_time_seconds

    ########################################## - CONVERT ANY FRAME_ID TO MM:SS
    @staticmethod
    def frame_to_timestamp(frame_index,DETECTION_FPS):
        """
        Convert a frame index to MM:SS timestamp.
        
        Args:
            frame_index (int): The current frame index.
            fps (int): Frames per second of the video.
            
        Returns:
            str: Time in MM:SS format.
        """
        total_seconds = frame_index / DETECTION_FPS  # Get time in seconds
        minutes = int(total_seconds // 60)  # Get minutes
        seconds = int(total_seconds % 60)  # Remaining seconds
        return f"{minutes:02}:{seconds:02}"
