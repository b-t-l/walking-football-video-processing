from data import CreateDatabase
from video_annotator import CreateVideoAnnotated
from reports import CreateDetectionsReport,CreateMatchReport,CreateOppositionReport
from team_images import ExtractTeamImages
from data import GetGameData
from view_transformer import CreateViewTransformations
from track_stitcher import TrackStitcher
from speed_and_distance_db_updater import SpeedDistanceUpdater
from posession_db_updater import BallPosessionUpdater
from utils import *
from team_assigner import Teams
import os

from video_annotator import AnnotatedVideoProcessor

def main():


    ############################################################ VARIABLES:
    GAME_ID = 22     # what game (id column) to process ?
    GAME_SOURCE = 'database'   # where the game details come from: 'excel' = games-logger.xlsx, 'database' = the game logger app (python -m game_logger)

    # def what to run: (manually adjust to suit )
    RUN_DETECTIONS = False       # Run yolo detections for players/goalkeeper/pitch/ball and save to database
    RUN_TRANSFORMATIONS = False  # calculate detection positions transformed to overhead view, team assignment, speed, distance 
    RUN_DETECTIONS_REPORT = True   # run report to check how performance of detections/transformation
    RUN_ANNOTATION_VIDEO_CREATOR = False  # create annotated video
    RUN_TEAM_IMAGES = False  # save pictures of the players from 10 random frames, sorted by what the team model says (to correct + retrain it)
    RUN_MATCH_REPORT = False  # create the 12-page match report (PDF): possession, territory, shape, distance, speed, running
    RUN_OPPOSITION_REPORT = False  # create the opposition reports (PDF, one per team): how they play, strong and weak points, game plan

    # DETECTION SETTINGS FOR VIDEO:
    PITCH_DETECTION_OVERRIDE = True    # do we want to overide pitch detection and rather temp provide the vertices from our logger file instead ?
    DURATION = 0   # How many seconds do we want to process from the GAME_RECORDS start time ? 960
    BATCH_DURATION = 2  # Process 1 second per batch
    DETECTION_FPS = 12  # Target FPS for detection (eg 12 = every 2nd frame in a 24 FPS video)
    EXCLUDE_TIME_RANGES = []  # Define the time ranges to exclude (in seconds) [(1, 2), (4, 5)]

    # COMMAND-LINE OPTIONS (the game logger's Run panel uses these; with none given, the values above are used as before):
    #   python main.py --game 22 --steps detections,transformations,annotation --duration 120
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--game', type=int, help='game id to process')
    _ap.add_argument('--steps', help='comma separated: detections, transformations, detection_report, match_report, opposition_report, annotation, team_images')
    _ap.add_argument('--duration', type=int, help='test run: only this many seconds from the game start; results are kept in separate test files')
    _args, _unknown = _ap.parse_known_args()
    if _args.game is not None:
        GAME_ID = _args.game
    if _args.steps is not None:
        _steps = {s.strip() for s in _args.steps.split(',') if s.strip()}
        _unknown_steps = _steps - {'detections', 'transformations', 'detection_report', 'match_report', 'opposition_report', 'annotation', 'team_images'}
        if _unknown_steps:
            raise SystemExit(f"STOPPED: unknown stage(s): {', '.join(sorted(_unknown_steps))}")
        RUN_DETECTIONS = 'detections' in _steps
        RUN_TRANSFORMATIONS = 'transformations' in _steps
        RUN_DETECTIONS_REPORT = 'detection_report' in _steps
        RUN_MATCH_REPORT = 'match_report' in _steps
        RUN_OPPOSITION_REPORT = 'opposition_report' in _steps
        RUN_ANNOTATION_VIDEO_CREATOR = 'annotation' in _steps
        RUN_TEAM_IMAGES = 'team_images' in _steps
    TEST_RUN = bool(_args.duration)
    if TEST_RUN:
        DURATION = _args.duration

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
    game = GetGameData(GAME_ID,GAME_DATA_FILE,source=GAME_SOURCE)
    GAME_RECORD = game.gameRecord()
    print('-------------------------- GAME DETAIL: ')
    for key, value in GAME_RECORD.items():
        print(f"{key}: {value}")
    
    GAME_PITCH_KEYPOINTS_WITH_NAMES = game.getGamePitchKeyPointsWithNames(GAME_RECORD)
    GAME_PITCH_BOUNDARIES = game.getGamePitchBoundaries(GAME_RECORD)
    GAME_PITCH_KEYPOINTS = game.getGamePitchKeyPoints(GAME_RECORD)

    # TEAMS:
    TEAMS_INSTANCE = Teams(source=GAME_SOURCE)
    ALL_TEAMS = TEAMS_INSTANCE.get_all_teams()
    TEAMS = TEAMS_INSTANCE.get_playing_teams([GAME_RECORD['team_a'], 
                                                     GAME_RECORD['team_b']])
    # WHERE EVERYTHING IS SAVED: when the game logger has a data folder set (e.g. on the external drive) the game gets its own
    # folder there: data/ (detections database, object images), reports/, videos/annotated/ and _work/ (throw-away files).
    # With no data folder set (or reading from the Excel file) the old shared output folder is used.
    GAME_PATHS = None
    if GAME_SOURCE == 'database':
        from game_logger import paths as game_paths
        try:
            GAME_PATHS = game_paths.pipeline_paths(GAME_ID)      # None = no data folder set
        except game_paths.GameFolderError as err:
            raise SystemExit(f"\nSTOPPED: {err}\n")
    if GAME_PATHS:
        DATABASE_FILE = GAME_PATHS['db']
        OBJECT_IMAGES_FOLDER = GAME_PATHS['object_images']
        OUTPUT_PATH = GAME_PATHS['reports']
        ANNOTATED_PATH = GAME_PATHS['annotated']
        WORK_DIR = GAME_PATHS['work']
        print(f"Game folder: {GAME_PATHS['folder']}")
    else:
        DATABASE_FILE = os.path.join(PARENT_DIR,f"output/game-id-{GAME_RECORD['game_id']}-{StringUtils.replace_spaces_with_underscores(GAME_RECORD['title'])}.db")
        ANNOTATED_PATH = OUTPUT_PATH
        WORK_DIR = OUTPUT_PATH
    if TEST_RUN:
        # a test run never touches the real results: it has its own database, object images and report folder
        _db_base, _db_ext = os.path.splitext(DATABASE_FILE)
        DATABASE_FILE = _db_base + '-test' + _db_ext
        OBJECT_IMAGES_FOLDER = OBJECT_IMAGES_FOLDER.rstrip('/\\') + '-test'
        OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'test')
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        print(f"TEST RUN: only the first {DURATION} seconds from the game start; results go to separate test files")
    print(f"Detections database: {DATABASE_FILE}")

    # the analysis video must exist before we start (it is found in the game folder by name unless a path is set on the game)
    if (RUN_DETECTIONS or RUN_ANNOTATION_VIDEO_CREATOR or RUN_TEAM_IMAGES) and not os.path.exists(str(GAME_RECORD['statistics_source_video'])):
        raise SystemExit(f"\nSTOPPED: the analysis video was not found: {GAME_RECORD['statistics_source_video']}\n"
                         "Put it in the game's videos folder as <game number>_analysis.mp4, or choose it on the game page in the game logger.\n")


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
                                GAME_PITCH_BOUNDARIES,
                                WORK_DIR=WORK_DIR)
        database.run()


    # - TRANSFORMATIONS: Do calculations for view transforming (to get positions in vertical/map view and calculate from it the speed/distance calculations:
    if RUN_TRANSFORMATIONS == True:
        db_transformations = CreateViewTransformations(DATABASE_FILE,
                                                       GAME_RECORD,
                                                       PITCH_DETECTION_OVERRIDE)
        db_transformations.run()
        # one stable player_id per real player (repairs broken tracker ids, settles team labels)
        db_track_stitcher = TrackStitcher(DATABASE_FILE,
                                          GAME_RECORD)
        db_track_stitcher.run()
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


    # - MATCH REPORT: 12-page PDF with team statistics for both teams
    if RUN_MATCH_REPORT == True:
        report = CreateMatchReport(DATABASE_FILE,
                                   OUTPUT_PATH,
                                   TEAMS,
                                   GAME_RECORD,
                                   DETECTION_FPS)
        report.run()

    # - OPPOSITION REPORTS: one scouting PDF per team, written for the other team's coach
    if RUN_OPPOSITION_REPORT == True:
        opposition = CreateOppositionReport(DATABASE_FILE,
                                            OUTPUT_PATH,
                                            TEAMS,
                                            GAME_RECORD,
                                            DETECTION_FPS)
        opposition.run()

    # - TEAM TRAINING IMAGES: players from 10 random frames saved as pictures, filed by what the team model said
    if RUN_TEAM_IMAGES == True:
        TEAM_IMAGES_FOLDER = os.path.join(GAME_PATHS['folder'] if GAME_PATHS else OUTPUT_PATH, 'team-images')
        ExtractTeamImages(DATABASE_FILE,
                          GAME_RECORD['statistics_source_video'],
                          TEAM_IMAGES_FOLDER,
                          GAME_RECORD,
                          n_frames=10).run()

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
                                GAME_PITCH_BOUNDARIES,
                                ANNOTATED_PATH=ANNOTATED_PATH,
                                WORK_DIR=WORK_DIR)
        annotated_video.run()


if __name__ == '__main__':
    main()