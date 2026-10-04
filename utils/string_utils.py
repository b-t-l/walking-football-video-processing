import numpy as np

class StringUtils:
    @staticmethod
    def capitalize_first_letter(string):
        return string.capitalize()

    @staticmethod
    def is_palindrome(string):
        return string == string[::-1]

    @staticmethod
    def replace_spaces_with_underscores(s):
        return s.replace(' ', '_')

    @staticmethod
    def string_vertices_to_nparray(GAME_RECORD,frame_id):

        # Convert to tuple
        left_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_bottom'].split(',')))
        left_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_left_top'].split(',')))
        right_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_top'].split(',')))
        right_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_right_bottom'].split(',')))
        centre_top_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_centre_top'].split(',')))
        centre_bottom_tuple = tuple(map(int, GAME_RECORD['pitch_vertices_centre_bottom'].split(',')))
        # now overide the [points from the detection keyframe with our own values:

        # Now you can use this tuple in a NumPy array
        array  = [{'frame_id': frame_id, 'name': 'left_bottom', 'x': left_bottom_tuple[0], 'y': left_bottom_tuple[1]}, 
                  {'frame_id': frame_id, 'name': 'left_top', 'x': left_top_tuple[0], 'y': left_top_tuple[1]}, 
                  {'frame_id': frame_id, 'name': 'centre_top', 'x': centre_top_tuple[0], 'y': centre_top_tuple[1]}, 
                  {'frame_id': frame_id, 'name': 'right_top', 'x': right_top_tuple[0], 'y': right_top_tuple[1]}, 
                  {'frame_id': frame_id, 'name': 'right_bottom', 'x': right_bottom_tuple[0], 'y': right_bottom_tuple[1]}, 
                  {'frame_id': frame_id, 'name': 'centre_bottom', 'x': centre_bottom_tuple[0], 'y': centre_bottom_tuple[1]}
                  ]

        return array
    