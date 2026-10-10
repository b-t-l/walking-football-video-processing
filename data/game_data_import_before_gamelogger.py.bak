import datetime
import pandas as pd

TIME_FIELDS = ('game_start_seconds', 'half_time_start_seconds', 'half_time_end_seconds', 'game_end_seconds')


def normalise_mmss(value):
    """
    Game-logger time cells are meant to be MM:SS (minutes:seconds into the video), but Excel quietly turns a typed
    "01:00" into a TIME value (1:00 = 1 hour 0 min) and an empty cell arrives as NaN. Return a clean 'MM:SS' string,
    or '' for an empty cell, so nothing downstream has to care how the cell was stored.
    Excel's hours are read as minutes and its minutes as seconds (what was typed is what is meant).
    Tip: formatting those cells as Text in Excel avoids the conversion altogether.
    """
    if value is None:
        return ''
    if isinstance(value, float) and pd.isna(value):
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, datetime.datetime):                  # e.g. 1900-01-02 14:30 when more than 24 'hours' were typed
        total_min = int((value - datetime.datetime(1899, 12, 30)).total_seconds() // 3600)
        seconds = value.minute
    elif isinstance(value, datetime.time):
        total_min, seconds = value.hour, value.minute
    elif isinstance(value, datetime.timedelta):
        total = int(value.total_seconds())
        total_min, seconds = total // 3600, (total % 3600) // 60
    elif isinstance(value, (int, float)):                     # Excel serial fraction of a day
        total = int(round(float(value) * 86400))
        total_min, seconds = total // 3600, (total % 3600) // 60
    else:
        return str(value).strip()
    return f"{total_min:02d}:{seconds:02d}"


## - read the game data xls so we can get titles,duration,teams etc
class GetGameData():

    def __init__(self, GAME_ID, GAME_DATA_FILE):

        print("-------------------------------------------------")
        print(f"Get game data: ID: {GAME_ID} from file : {GAME_DATA_FILE}")

        self.GAME_DATA_FILE = GAME_DATA_FILE
        self.GAME_ID = GAME_ID
        self.KEYPOINT_NAMES = [
                "left_bottom",
                "left_top",
                "centre_top",
                "right_top",
                "right_bottom",
                "centre_bottom"
                ]
        self.KEYPOINTS_SKELETON = [
                [1,5],
                [5,4],
                [4,2],
                [2,1]
                ]

    ########################################## - GAME RECORD: read the excel file and get the row for the matching game_id
    def gameRecord(self):

        # Read the XLS file
        df = pd.read_excel(self.GAME_DATA_FILE)

        # Convert to a list of dictionaries
        data = df.to_dict(orient="records")

        # Find the first row where game_id is 0
        matching_record = next((row for row in data if row['game_id'] == self.GAME_ID), None)

        # time cells -> clean 'MM:SS' strings ('' when empty)
        if matching_record is not None:
            for field in TIME_FIELDS:
                if field in matching_record:
                    matching_record[field] = normalise_mmss(matching_record[field])

        return matching_record

    ########################################## - GAME PITCH KEYPOINTS: get the pitch key points from the game record
    def getGamePitchKeyPointsWithNames(self,GAME_RECORD):

        # Convert to tuple
        left_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_bottom'].split(',')))
        left_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_top'].split(',')))
        right_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_top'].split(',')))
        right_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_bottom'].split(',')))
        centre_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_centre_top'].split(',')))
        centre_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_centre_bottom'].split(',')))
        # now overide the [points from the detection keyframe with our own values:

        # Now you can use this tuple in a NumPy array
        array  = [{'name': 'left_bottom', 'x': left_bottom_tuple[0], 'y': left_bottom_tuple[1]}, 
                  {'name': 'left_top', 'x': left_top_tuple[0], 'y': left_top_tuple[1]}, 
                  {'name': 'centre_top', 'x': centre_top_tuple[0], 'y': centre_top_tuple[1]}, 
                  {'name': 'right_top', 'x': right_top_tuple[0], 'y': right_top_tuple[1]}, 
                  {'name': 'right_bottom', 'x': right_bottom_tuple[0], 'y': right_bottom_tuple[1]}, 
                  {'name': 'centre_bottom', 'x': centre_bottom_tuple[0], 'y': centre_bottom_tuple[1]}
                  ]

        return array

    ########################################## - GAME PITCH KEYPOINTS: get the pitch key points from the game record
    def getGamePitchKeyPoints(self,GAME_RECORD):

        # Convert to tuple
        left_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_bottom'].split(',')))
        left_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_top'].split(',')))
        right_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_top'].split(',')))
        right_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_bottom'].split(',')))
        centre_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_centre_top'].split(',')))
        centre_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_centre_bottom'].split(',')))
        # now overide the [points from the detection keyframe with our own values:

        # Now you can use this tuple in a NumPy array
        array  = [(left_bottom_tuple[0],left_bottom_tuple[1]), 
                  (left_top_tuple[0],left_top_tuple[1]), 
                  (centre_top_tuple[0],centre_top_tuple[1]), 
                  (right_top_tuple[0],right_top_tuple[1]), 
                  (right_bottom_tuple[0],right_bottom_tuple[1]), 
                  (centre_bottom_tuple[0],centre_bottom_tuple[1])
                  ]

        return array
    
    ########################################## - GAME PITCH BOUNDARIES: get the corners of the pitch 
    def getGamePitchBoundaries(self,GAME_RECORD):

        # Convert to tuple
        left_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_bottom'].split(',')))
        left_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_top'].split(',')))
        right_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_top'].split(',')))
        right_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_bottom'].split(',')))

        return [
            left_bottom_tuple,
            left_top_tuple,
            right_top_tuple,
            right_bottom_tuple
        ]