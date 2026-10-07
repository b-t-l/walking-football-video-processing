import pandas as pd

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