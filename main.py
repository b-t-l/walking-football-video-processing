from data import CreateDatabase
from video_annotator import CreateVideoAnnotated
from reports import CreateStatisticsReport,CreateDetectionsReport
from data import GetGameData
from view_transformer import CreateViewTransformations
from speed_and_distance_db_updater import SpeedDistanceUpdater
from posession_db_updater import BallPosessionUpdater
from utils import *
from team_assigner import Teams
import os

from video_annotator import AnnotatedVideoProcessor

def main():


    ############################################################ VARIABLES:
    GAME_ID = 22     # what game (id column) to process ?

    # def what to run: (manually adjust to suit )
    RUN_DETECTIONS = False       # Run yolo detections for players/goalkeeper/pitch/ball and save to database
    RUN_TRANSFORMATIONS = True  # calculate detection positions transformed to overhead view, team assignment, speed, distance 
    RUN_DETECTIONS_REPORT = False   # run report to check how performance of detections/transformation
    RUN_ANNOTATION_VIDEO_CREATOR = True  # create annotated video
    RUN_MATCH_REPORT_CREATOR = True  # create match report/statistics

    # DETECTION SETTINGS FOR VIDEO:
    PITCH_DETECTION_OVERRIDE = True    # do we want to overide pitch detection and rather temp provide the vertices from our logger file instead ?
    DURATION = 0   # How many seconds do we want to process from the GAME_RECORDS start time ? 960
    BATCH_DURATION = 2  # Process 1 second per batch
    DETECTION_FPS = 24  # Target FPS for detection (eg 12 = every 2nd frame in a 24 FPS video)
    EXCLUDE_TIME_RANGES = []  # Define the time ranges to exclude (in seconds) [(1, 2), (4, 5)]

    # INPUT GAME: Set which game to read in from with its details stored in the game-logger.xls
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(PARENT_DIR)
    def find_file_in_parent(file_name, start_dir):
        for root, dirs, files in os.walk(start_dir):
            if file_name in files:
                return os.path.join(root, file_name)
        return None
    GAME_DATA_FILE = find_file_in_parent('games-logger.xlsx', PARENT_DIR)
    GAME_RECORD = None

    # DATABASE/IMAGE STORAGE
    DATABASE_FILE = None     # database we will create:
    OBJECT_IMAGES_FOLDER = os.path.join(os.getcwd(), 'images_object_detection')   # where will we store all the bbox images of players from each frame

    # STORE OUTPUT TO:
    OUTPUT_PATH = os.path.join(PARENT_DIR, 'output')

    
    ############################################################ RUN SCRIPTS:

    # - GET GAME DETAILS: details of the game to process from the game logger xls and update necessary variables:
    game = GetGameData(GAME_ID,GAME_DATA_FILE)
    GAME_RECORD = game.gameRecord()
    print('-------------------------- GAME DETAIL: ')
    for key, value in GAME_RECORD.items():
        print(f"{key}: {value}")
    
    GAME_PITCH_KEYPOINTS_WITH_NAMES = game.getGamePitchKeyPointsWithNames(GAME_RECORD)
    GAME_PITCH_BOUNDARIES = game.getGamePitchBoundaries(GAME_RECORD)
    GAME_PITCH_KEYPOINTS = game.getGamePitchKeyPoints(GAME_RECORD)

    # TEAMS:
    TEAMS_INSTANCE = Teams()
    ALL_TEAMS = TEAMS_INSTANCE.get_all_teams()
    TEAMS = TEAMS_INSTANCE.get_playing_teams([GAME_RECORD['team_a'], 
                                                     GAME_RECORD['team_b']])
    # DATABASE:
    DATABASE_FILE = os.path.join(PARENT_DIR,f"output/game-id-{GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(GAME_RECORD['title'])}.db")


    print('-------------------------- PROCESSING STARTED: ')

    # - DETECTIONS SAVE TO DB: run YOLO detections and team assigner then store to db
    if RUN_DETECTIONS == True:
        database = CreateDatabase(DURATION, 
                                BATCH_DURATION, 
                                DETECTION_FPS, 
                                EXCLUDE_TIME_RANGES, 
                                DATABASE_FILE, 
                                OBJECT_IMAGES_FOLDER,
                                TEAMS,
                                GAME_RECORD,
                                PITCH_DETECTION_OVERRIDE,
                                GAME_PITCH_KEYPOINTS_WITH_NAMES,
                                GAME_PITCH_KEYPOINTS,
                                GAME_PITCH_BOUNDARIES)
        database.run()


    # - TRANSFORMATIONS: Do calculations for view transforming (to get positions in vertical/map view and calculate from it the speed/distance calculations:
    if RUN_TRANSFORMATIONS == True:
        db_transformations = CreateViewTransformations(DATABASE_FILE,
                                                       GAME_RECORD,
                                                       PITCH_DETECTION_OVERRIDE)
        db_transformations.run()
        db_speed_distance = SpeedDistanceUpdater(DATABASE_FILE,
                                                 GAME_RECORD,
                                                 DURATION)
        db_speed_distance.run()
        db_ball_posession = BallPosessionUpdater(DATABASE_FILE,
                                                 GAME_RECORD)
        db_ball_posession.run()


    # - DETECTION REPORT: create report with detection statistics
    if RUN_DETECTIONS_REPORT == True:
        report = CreateDetectionsReport(DATABASE_FILE,
                                        OUTPUT_PATH,
                                        TEAMS,
                                        GAME_RECORD,
                                        DETECTION_FPS)
        report.run()


    # - MATCH REPORT: Create report with match statistics
    if RUN_MATCH_REPORT_CREATOR == True:
        report = CreateStatisticsReport(DATABASE_FILE,
                                        OUTPUT_PATH,
                                        TEAMS,
                                        GAME_RECORD,
                                        DETECTION_FPS)
        report.run()

    # - ANNOTATED VIDEO: run creating an annotated video with player detections, pitch top view etc
    if RUN_ANNOTATION_VIDEO_CREATOR == True:
        
        annotated_video = CreateVideoAnnotated(DURATION, 
                                BATCH_DURATION, 
                                DATABASE_FILE, 
                                OBJECT_IMAGES_FOLDER,
                                OUTPUT_PATH,
                                TEAMS,
                                GAME_RECORD,
                                PITCH_DETECTION_OVERRIDE,
                                GAME_PITCH_KEYPOINTS_WITH_NAMES,
                                GAME_PITCH_BOUNDARIES)
        annotated_video.run()


if __name__ == '__main__':
    main()