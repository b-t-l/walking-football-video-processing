import cv2
import numpy as np
import ast
from collections import defaultdict
from utils import *

PLAYER_COLOR = (255, 255, 0)
PLAYER_TEXT_COLOR = (0, 0, 0)
REFEREE_COLOR = (96, 136, 250)
REFEREE_TEXT_COLOR = (0,60,255)
GOALKEEPER_COLOR = (232,224,132)
GOALKEEPER_TEXT_COLOR = (173,152,37)
BALL_COLOR = (255,255,255)

STATS_BOX_WIDTH = 3820

# Bottom-panel layout was drawn for a 3840x2160 frame. Scale it onto the frame being annotated.
REF_FRAME_WIDTH = 3840
REF_FRAME_HEIGHT = 2160
REF_PANEL_TOP = 1300


def frame_layout_scale(frame):
    """Scale factor that fits the 4K overlay layout inside this frame."""
    height, width = frame.shape[:2]
    scale = min(width / REF_FRAME_WIDTH, height / REF_FRAME_HEIGHT)
    return scale, width, height


############################################### DRAW ALL ANNOTATIONS: Draw all the annotations onto the video we want:
def draw_annotations_batch(frame_ids, frames, data_batch, pitch_points_batch, TEAMS, PITCH_DETECTION_OVERRIDE, DETECTION_FPS,ORIGINAL_VIDEO_FPS,GAME_RUNNING_SPEED, PITCH_LENGTH_M=46.0, PITCH_WIDTH_M=21.0, PITCH_OVERLAY=None):
    """Process multiple frames in a batch for better performance."""
    annotation_frames = []

    frame_height, frame_width = frames[0].shape[:2]
    scale, _, _ = frame_layout_scale(frames[0])
    panel_top = min(int(REF_PANEL_TOP * scale), frame_height)

    # Process frames in batch
    for frame_id, frame, data, pitch_points in zip(frame_ids, frames, data_batch, pitch_points_batch):
        # Use numpy operations where possible
        annotation_frame = frame.copy()

        # Black panel under the pitch map. Rows are [y1:y2], columns are [x1:x2].
        annotation_frame[panel_top:frame_height, 0:frame_width] = 0
        
        
        # Apply annotations (consider vectorizing these operations)
        annotation_frame = draw_pitch_detection_on_field(annotation_frame, frame_id, pitch_points)
        annotation_frame = draw_object_detections_on_field(annotation_frame, frame_id, data, pitch_points,GAME_RUNNING_SPEED)
        annotation_frame = draw_pitch_top_view(annotation_frame, data, pitch_points, PITCH_LENGTH_M, PITCH_WIDTH_M, PITCH_OVERLAY)
        annotation_frame = draw_stats(annotation_frame, data)
        annotation_frame = draw_video_details(annotation_frame, frame_id, data, DETECTION_FPS,ORIGINAL_VIDEO_FPS)
        
        annotation_frames.append(annotation_frame)
    
    return annotation_frames

# Original function maintained for backward compatibility
def draw_annotations(frame_id, frame, data, pitch_points, TEAMS, PITCH_DETECTION_OVERRIDE, DETECTION_FPS):
    """Wrapper for single frame processing"""
    return draw_annotations_batch(
        [frame_id], 
        [frame], 
        [data], 
        [pitch_points], 
        TEAMS, 
        PITCH_DETECTION_OVERRIDE, 
        DETECTION_FPS
    )[0]


def is_point_left_or_right(pitch_keypoints, obj_x, obj_y):
    """
    Determine if a point (obj_x, obj_y) is left or right of the line 
    running from centre_top to centre_bottom.
    
    Args:
        pitch_keypoints (list): List of keypoints with 'name', 'x', 'y'.
        obj_x (int): X-coordinate of the object.
        obj_y (int): Y-coordinate of the object.
    
    Returns:
        str: 'left', 'right', or 'on the line'
    """
    # Extract centre_top and centre_bottom from the keypoints
    centre_top = next(kp for kp in pitch_keypoints if kp['name'] == 'centre_top')
    centre_bottom = next(kp for kp in pitch_keypoints if kp['name'] == 'centre_bottom')
    
    x1, y1 = centre_top['x'], centre_top['y']
    x2, y2 = centre_bottom['x'], centre_bottom['y']
    
    # Calculate cross product
    cross_product = (x2 - x1) * (obj_y - y1) - (y2 - y1) * (obj_x - x1)
    
    if cross_product > 0:
        return 'left'
    elif cross_product < 0:
        return 'right'
    else:
        return 'centre'
    
##################################### -- draws the predicted pitch outline on the video
def draw_pitch_detection_on_field(frame,frame_id,pitch_points):

    overlay = frame.copy()

    # Draw the pitch keypoints:
    for i in range(len(pitch_points)):
        
        # Extract details of the current keypoint
        #frame_id = pitch_points[i]['frame_id']
        name = pitch_points[i]['name']
        x = pitch_points[i]['x']
        y = pitch_points[i]['y']
        point = (int(x), int(y))

        # Draw a circle for the keypoint
        cv2.circle(overlay, point, 15, (0, 255, 0), -1)  # Red point

        # Add the name text below the circle
        text_position = (point[0], point[1] + 25)  # Slightly below the circle
        cv2.putText(
            frame,
            name,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font size
            (255, 255, 255),  # White color
            1,  # Thickness
            cv2.LINE_AA,
        )

        # Draw a line to the next keypoint (cyclically connecting the last to the first)
        next_index = (i + 1) % len(pitch_points)  # Wrap around to form a loop
        next_point = (int(pitch_points[next_index]['x']), int(pitch_points[next_index]['y']))
        cv2.line(overlay, point, next_point, (0, 255, 0), 2)  # Green line

    # Blend the overlay with the original frame using transparency
    alpha = 0.4  # Set the transparency level (0.0 to 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend the overlay with the original frame

    return frame



############################################### DRAW PLAYER ANNOTATION ON FRAME VIDEO
def draw_object_detections_on_field(frame,frame_id,data,pitch_points,GAME_RUNNING_SPEED):

    # Get the centre top pitch point ( we will use to place the annoation for each player/goalkeeper/ref on left or side of centre)
    #centre_top = next((kp for kp in pitch_points if kp['name'] == 'centre_top'), None)
    # just draw out in out points:
    # Define the points

    '''
    polygon_points = [(297, 1104), (1005, 217), (1913, 217), (2868, 199), (3624, 1122), (1913, 1096)]

    # Draw red circles at each point
    radius = 10
    color = (0, 0, 255)  # Red in BGR
    thickness = -1  # Filled circle

    for point in polygon_points:
        cv2.circle(frame, point, radius, color, thickness)

    # Draw lines connecting the points
    line_color = (255, 255, 255)  # White lines
    line_thickness = 2

    for i in range(len(polygon_points)):
        # Get the current point and the next point (wrap around for the last point)
        start_point = polygon_points[i]
        end_point = polygon_points[(i + 1) % len(polygon_points)]  # Wrap around to the first point
        cv2.line(frame, start_point, end_point, line_color, line_thickness)
    '''


    ########## 1. draw the tranparent items we want:

    # Create an overlay for the object detections as we want all this dat to be slightly transparent.
    overlay = frame.copy()  # Create a copy of the frame for the overlay

    # Draw the transformed points of players on the video:
    for object in data:
        
        # skip doing anything if it is a ball of referee:
        if object['class_name'] == "ball" or object['team'] == "Goalkeeper" or object['team'] == "Referee":
            continue
        
        #### Setup the variables:
        x_min =  int(object['xmin'])
        y_min =  int(object['ymin'])
        x_max =  int(object['xmax'])
        y_max =  int(object['ymax'])
        tracker_id = int(object["tracker_id"])
        team = object['team']
        team_color = object['team_color']
        speed = object['speed_km_per_hour']
        if speed is None:
            speed = 0
        distance = object['total_distance_metres']

        # work out if the object is on the left/right/center of the pitch from the centre of the bbox:
        object_placement = is_point_left_or_right(pitch_points, ((x_max-x_min)/2 + x_min), ((y_max-y_min)/2 + y_min))
        
        #### color:
        color = PLAYER_COLOR
        text_color = PLAYER_TEXT_COLOR
        if object["team_color"]:
            color = ast.literal_eval(object["team_color"])
            text_color = get_contrasting_color(color)
        
        ##### draw the circle showing the foot position and team color:
        '''
        if object_placement == 'left' and object['class_name'] != "ball":
            # Draw the circle
            cv2.circle(overlay, (x_max,y_max+4), 8, color, -1)
        elif object_placement == 'right' and object['class_name'] != "ball":
            cv2.circle(overlay, (x_min,y_max+4), 8, color, -1)
        '''
        ##### draw the plus sign showing the foot position and team color:
        if object_placement == 'left' and object['class_name'] != "ball":
            # Calculate the center of the plus sign
            center_x, center_y = x_max, y_max + 4
        elif object_placement == 'right' and object['class_name'] != "ball":
            center_x, center_y = x_min, y_max + 4
        # Draw the plus sign
        line_length = 8  # Length of each line in the plus sign
        line_thickness = 2  # Thickness of the lines
        # Horizontal line
        cv2.line(overlay, (center_x - line_length, center_y), (center_x + line_length, center_y), color, line_thickness)
        # Vertical line
        cv2.line(overlay, (center_x, center_y - line_length), (center_x, center_y + line_length), color, line_thickness)


        ### Draw the line for team
        '''
        y_start_drawing = int(y_max+3)
        x_start_drawing = x_min
        line_thickness = 2
        line_length = min(((x_max - x_min) // 2),80)
        line_start = (x_start_drawing, y_start_drawing)
        line_end = (x_start_drawing + line_length, y_start_drawing)
        if object_placement == 'left':
            line_start = (x_max - line_length, y_start_drawing)
            line_end = (x_max, y_start_drawing)
        elif object_placement == 'right':
            line_start = (x_start_drawing, y_start_drawing)
            line_end = (x_start_drawing + line_length, y_start_drawing)
        cv2.line(overlay, line_start, line_end, color, line_thickness)
        '''

        ##### speed color: determine a speed based on relative to GAME_RUNNING_SPEED and show different color for the speed green,orange, red ( eg 0-5kmh = green, 5-10kmh = dark green, 10-15:orange , >15kmh = red )
        
        # we have 3 distinctions for running colors:
        game_running_segment = GAME_RUNNING_SPEED/3
        speed_color = (0, 255, 0)
        speed_line_length = 10
        max_line_length = 80
        # Determine speed color and line length based on speed relative to GAME_RUNNING_SPEED
        if speed is not None and distance is not None:
            if speed <= 0:
                speed_color = (0, 0, 0)  # Green for 0-5 km/h
                speed_line_length = max(min(max_line_length, int(0.3 * speed)), 10)  # 30% of speed, minimum 10 pixels
            elif speed <= 10:
                speed_color = (0, 0, 0)  # Dark green for 5-10 km/h
                speed_line_length = max(min(max_line_length, int(0.4 * speed)), 10)  # 40% of speed, minimum 10 pixels
            elif speed <= 15:
                speed_color = (0, 165, 255)  # Orange for 10-15 km/h
                speed_line_length = max(min(max_line_length, int(0.5 * speed)), 10)  # 50% of speed, minimum 10 pixels
            else:
                speed_color = (0, 0, 255)  # Red for >15 km/h
                speed_line_length = max(min(max_line_length, int(0.6 * speed)), 10)  # 60% of speed, minimum 10 pixels
        
            speed_line_length = max(min(max_line_length, speed_line_length), 10)  # Ensure minimum length
            #speed_line_length = 40
        else:
            speed_line_length = 10  # Set line length to 0 if speed is None

        #print(f"speed: {speed} - {speed_line_length}")
        
        ### Draw a bounding box around the player with different color for running:
        if speed > 10:
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), speed_color, 4)
        else:
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 1) 
        
        ### add the bounding box area as text for debugging:
        current_bbox_area = (x_max - x_min) * (y_max - y_min)


        ### show the tracker id + speed of player (show speed only if running )
        player_id = f"{str(int(object['tracker_id']))}"
        confidence = object['confidence']
        player_text_ypos = int(y_max) + 10
        player_text_xpos = int(x_min) + 10
        if object_placement == 'left':
            player_text_xpos = int(x_max) - 50
        # put player id at foot position:
        cv2.putText(overlay, f"{speed:.2f}km/h",(player_text_xpos,player_text_ypos+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),2)
        
        # add speed if player is running:
        if speed is not None and speed > 10:
            cv2.putText(overlay, f"{player_id}-{speed:.2f}km/h",(x_min,y_min-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2)  

        ### show the confidence of the prediction:



        #### DRAW line underneath for team and line for speed they are travelling:
        #speed_y_start_drawing = int(y_max-5)
        #speed_x_start_drawing = x_min
        #speed_line_thickness = 2
        #speed_line_start = (speed_x_start_drawing, speed_y_start_drawing)
        #speed_line_end = (speed_x_start_drawing + speed_line_length, speed_y_start_drawing)

        #cv2.line(overlay, speed_line_start, speed_line_end, color, line_thickness)


    # Blend the overlay with the original frame using transparency
    alpha = 0.6  # Set the transparency level (0.0 to 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend the overlay with the original frame


    ######################### 2: Draw fully opaque elements directly on the frame: BALL AND POSESSION
    for object in data:

        # Example of drawing a fully opaque bounding box
        x_min = int(object['xmin'])
        y_min = int(object['ymin'])
        x_max = int(object['xmax'])
        y_max = int(object['ymax'])

        # only draw if it is a ball
        if object['class_name'] == "ball":

            y= int(y_min)
            x,_ = get_center_of_bbox(x_min,y_min,x_max,y_max)
            triangle_points = np.array([
                [x,y],
                [x-10,y-20],
                [x+10,y-20],
            ])
            cv2.drawContours(frame, [triangle_points],0,BALL_COLOR, cv2.FILLED)

        # draw an ellipse if they are in posession of the ball:
        if object['ball_posession'] == True:

            posession_color = (0, 0, 255)  # Red color in BGR format
            y= int(y_min)
            x,_ = get_center_of_bbox(x_min,y_min,x_max,y_max)
            triangle_points = np.array([
                [x,y],
                [x-10,y-20],
                [x+10,y-20],
                ])
            cv2.drawContours(frame, [triangle_points],0,posession_color, cv2.FILLED)    

    return frame



##################################### -- draws the predictedtop view pitch with players/ball position
def draw_pitch_top_view(frame,data,pitch_points, pitch_length_m=46.0, pitch_width_m=21.0, overlay=None):

    # Real-world rectangle dimensions for THIS game (legacy default 46x21 kept only for
    # backward compatibility with callers that don't pass a per-game size).
    # NOTE: "pitch_width" here is actually the length axis (goal-to-goal) and
    # "pitch_height" is the width axis (touchline-to-touchline) -- naming kept as-is
    # to minimise the diff against the rest of this function.
    pitch_width = pitch_length_m  # meters
    pitch_height = pitch_width_m  # meters

    # Pitch map size and position on a 3840x2160 frame, then scaled to this frame.
    new_width = 460  # pixels
    new_height = 210  # pixels
    scaling_factor = 3.6
    scale, frame_width, frame_height = frame_layout_scale(frame)

    scaled_width = max(1, int(new_width * scaling_factor * scale))
    scaled_height = max(1, int(new_height * scaling_factor * scale))
    if overlay:
        # Calibrated game: draw the inset at the pitch's TRUE aspect ratio (width/length) so a
        # circle stays a circle. The fit-to-frame block below shrinks it uniformly if needed.
        scaled_height = max(1, int(scaled_width * pitch_height / pitch_width))
    x_offset = int(50 * scale)
    y_offset = int(1350 * scale)

    max_width = max(1, frame_width - x_offset)
    max_height = max(1, frame_height - y_offset)
    if scaled_width > max_width or scaled_height > max_height:
        fit = min(max_width / scaled_width, max_height / scaled_height)
        scaled_width = max(1, int(scaled_width * fit))
        scaled_height = max(1, int(scaled_height * fit))
    x_offset = min(x_offset, frame_width - scaled_width)
    y_offset = min(y_offset, frame_height - scaled_height)

    # Function to transform points with scaling
    def transform_point_to_new_rectangle(x, y, pitch_width, pitch_height, scaled_width, scaled_height, x_offset, y_offset):
        # World coords from the fitted per-video transformer are centred on the pitch
        # middle (0,0), not a corner -- shift into 0..pitch_width / 0..pitch_height
        # before scaling into the inset rectangle. (Harmless no-op under the old
        # corner-origin convention would have needed its own un-shifted values, but
        # that convention is no longer produced anywhere in this pipeline.)
        x_shifted = x + (pitch_width / 2.0)
        y_shifted = y + (pitch_height / 2.0)
        x_scaled = (x_shifted / pitch_width) * scaled_width
        y_scaled = (y_shifted / pitch_height) * scaled_height
        x_final = int(x_scaled + x_offset)
        y_final = int(y_scaled + y_offset)
        return x_final, y_final

    # Load the background image
    if overlay:
        # plain grass colour: pitch_map.jpg has its own pre-drawn 46x21 markings which would
        # confuse an accuracy test (our own ideal markings are drawn below instead)
        frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = (46, 96, 46)
    else:
        background_img = cv2.imread("images/pitch_map.jpg")  # Replace with your background image
        if background_img is None:
            raise FileNotFoundError("The background image could not be loaded. Check the file path.")

        # Resize the background image to match the scaled rectangle dimensions
        background_img_resized = cv2.resize(background_img, (scaled_width, scaled_height))

        # Add the background image to the frame at the specified position
        frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = background_img_resized

    # Draw the scaled rectangle on the existing frame
    cv2.rectangle(
        frame,
        (x_offset, y_offset),
        (x_offset + scaled_width, y_offset + scaled_height),
        (255, 255, 255),
        2  # White rectangle
    )

    
    ## ACCURACY TEST OVERLAY (optional): ideal centre circle / goal-end arcs (magenta) and where the
    ## clicked calibration circle points actually land after the fitted transform (yellow dots).
    ## If the fit were perfect every yellow dot would sit exactly on a magenta line.
    def world_to_inset(wx, wy):
        px, py = transform_point_to_new_rectangle(wx, wy, pitch_width, pitch_height, scaled_width, scaled_height, x_offset, y_offset)
        return px, (y_offset + scaled_height) - (py - y_offset)

    if overlay:
        line_w = max(1, int(3 * scale))
        # halfway line
        cv2.line(frame, world_to_inset(0, -pitch_height / 2), world_to_inset(0, pitch_height / 2), (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
        for shape in overlay.get('shapes', []):
            shape_pts = np.array([world_to_inset(wx, wy) for wx, wy in shape['pts']], dtype=np.int32)
            cv2.polylines(frame, [shape_pts.reshape(-1, 1, 2)], False, (255, 0, 255), line_w, cv2.LINE_AA)
        for pt in overlay.get('points', []):
            c = world_to_inset(pt['x'], pt['y'])
            cv2.circle(frame, c, max(3, int(11 * scale)), (0, 0, 0), -1)
            cv2.circle(frame, c, max(2, int(8 * scale)), (0, 255, 255), -1)

    ## Draw the transformed pitch points and connecting lines from the pitch_points inputted on the offest and scaled map
    # Transform and draw pitch points
    transformed_pitch_points = []
    for point in pitch_points:
        transformed_point = transform_point_to_new_rectangle(
            point["x_transformed_metres"],
            point["y_transformed_metres"],
            pitch_width,
            pitch_height,
            scaled_width,
            scaled_height,
            x_offset,
            y_offset
        )
        # Invert the y-coordinate to match the pitch's orientation
        transformed_point = (transformed_point[0], (y_offset + scaled_height) - (transformed_point[1] - y_offset))
        transformed_pitch_points.append(transformed_point)

        # Draw the point on the frame
        cv2.circle(frame, transformed_point, max(1, int(8 * scale)), (0, 255, 0), -1)  # Green dots for pitch points
    
    # Connect pitch points with lines
    for i in range(len(transformed_pitch_points)):
        start_point = transformed_pitch_points[i]
        end_point = transformed_pitch_points[(i + 1) % len(transformed_pitch_points)]  # Loop back to the start
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # Yellow lines for pitch edges
  

    ## Draw the transformed points of players on the top view pitch:
    for object in data:

        player_circle_radius = max(4, int(25 * scale))

        # - draw the player:
        if object["x_transformed_metres"] and object["y_transformed_metres"]:

            map_position = transform_point_to_new_rectangle(float(object["x_transformed_metres"]), 
                                                                float(object["y_transformed_metres"]), 
                                                                pitch_width, 
                                                                pitch_height, 
                                                                scaled_width, 
                                                                scaled_height,
                                                                x_offset,
                                                                y_offset)
            # invert the y pos :
            final_position_y = (y_offset + scaled_height) - ( map_position[1] - y_offset )   # dot centred on the foot position (was lifted by its radius)
            # create the final position to use in drawing the circle ( include the diameter offset because of size )
            

            # draw a circle representing the player/referee/goalkeeper
            if object['class_name'] != "ball" and object['class_name'] != "referee":
                
                # deal with colors
                circle_color = PLAYER_COLOR
                circle_text_color = PLAYER_TEXT_COLOR
                if object["team_color"]:
                    circle_color = ast.literal_eval(object["team_color"])
                    circle_text_color = get_contrasting_color(circle_color)
                
                # draw a slightly bigger red circle if they are in posession of the ball:
                if object['ball_posession'] == True:
                    cv2.circle(frame, (map_position[0], final_position_y), player_circle_radius + max(1, int(5 * scale)), (0,0,255), -1)

                # draw the player circle + tracker_id text
                cv2.circle(frame, (map_position[0], final_position_y), player_circle_radius, circle_color, -1)
                #text_position = (map_position[0]-7,final_position_y+4)  # Slightly below the circle
                #cv2.putText(
                #    frame,
                #    str(int(object['tracker_id'])),
                #    text_position,
                #    cv2.FONT_HERSHEY_SIMPLEX,
                #    0.5,
                #    circle_text_color,
                #    2,
                #    cv2.LINE_AA,
                #)

            
            # draw a circle representing the ball
            if object['class_name'] == "ball":
                cv2.circle(frame, (map_position[0], final_position_y), max(2, player_circle_radius // 2), BALL_COLOR, -1) 
                
    return frame


############################################### DRAW STATS NEXT TO THE PTICH TOP VIEW: TEAM/NR OF PLAYERS
def draw_stats(frame, data, padding_top=20, padding_bottom=20, font_scale=1.5):
    
    # Dictionary to count occurrences of teams and their colors
    team_data = defaultdict(lambda: {'team_color': None, 'nr_of_players': 0})

    # Process the data
    for entry in data:
        team = entry['team']
        team_color = entry['team_color']
        
        # Skip Goalkeeper and Referee teams, only count players
        if team not in ['Goalkeeper', 'Referee'] and entry['class_name'] == 'player':
            team_data[team]['nr_of_players'] += 1
            if team_data[team]['team_color'] is None:
                team_data[team]['team_color'] = team_color

    # Convert defaultdict to a regular list of dictionaries and sort alphabetically
    team_results = sorted(
        [{'team': team, 'team_color': info['team_color'], 'nr_of_players': info['nr_of_players']}
         for team, info in team_data.items()],
        key=lambda x: x['team']
    )
    
    # Team statistics sit to the right of the pitch map. Positions are for 3840x2160.
    scale, _, _ = frame_layout_scale(frame)
    offset_x = int(1800 * scale)
    y_offset = int(1400 * scale)
    padding_top = max(2, int(padding_top * scale))
    padding_bottom = max(2, int(padding_bottom * scale))
    font_scale = font_scale * scale
    text_thickness = max(1, int(4 * scale))
    box_padding = max(4, int(20 * scale))
    text_pad = max(2, int(10 * scale))
        
    for team in team_results:

        team_name = f"{team['team']}: {team['nr_of_players']} players"
        team_color = ast.literal_eval(team["team_color"])
        contrast_color = get_contrasting_color(team_color)

        # Calculate the size of the text based on the dynamic font scale
        text_size, _ = cv2.getTextSize(team_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        text_width, text_height = text_size
        
        # Adjust the height of the rectangle with padding
        rect_height = text_height + padding_top + padding_bottom
            
        # Draw background rectangle with padding
        cv2.rectangle(frame, (offset_x, y_offset), (offset_x + text_width + box_padding, y_offset + rect_height), team_color, -1)
            
        # Draw the text over the rectangle, adjusting for padding
        cv2.putText(frame, team_name, (offset_x + text_pad, y_offset + text_height + padding_top), cv2.FONT_HERSHEY_SIMPLEX, font_scale, contrast_color, text_thickness)
            
        # Update y_offset for the next team
        y_offset += rect_height + max(4, int(20 * scale))

    return frame



############################################### DRAW VIDEO STATS ON THE RIGHT SIDE:
def draw_video_details(frame,frame_id,data,DETECTION_FPS,ORIGINAL_VIDEO_FPS):

    scale, _, _ = frame_layout_scale(frame)
    offset_x = int(3000 * scale)
    y_offset = int(1400 * scale)
    font_scale = 1 * scale
    thickness = max(1, int(2 * scale))
    line_gap = max(16, int(50 * scale))

    text = f"Frame id: {str(frame_id)}"
    cv2.putText(frame, text, (offset_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

    detection_fps_txt = f"Detection fps: {DETECTION_FPS}"
    cv2.putText(frame, detection_fps_txt, (offset_x, y_offset + line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

    video_position_time = TimeUtils.frame_to_timestamp(frame_id,ORIGINAL_VIDEO_FPS)
    cv2.putText(frame, video_position_time, (offset_x, y_offset + line_gap * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

    return frame



############################################### GENERAL REUSED FUNCTIONS
def get_contrasting_color(bgr_color):
    """
    Compute a contrasting text color (black or white) for the given BGR background color.
    
    Args:
        bgr_color (tuple): Background color in BGR format (B, G, R).
    
    Returns:
        tuple: Contrasting color in BGR format.
    """
    b, g, r = bgr_color
    # Calculate luminance (perceived brightness)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Choose black for bright backgrounds, white for dark backgrounds
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

def get_center_of_bbox(x1,y1,x2,y2):
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(x_min,y_min,x_max,y_max):
    return x_max - x_min

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(x1,y1,x2,y2):
    return int((x1+x2)/2),int(y2)