import cv2
import numpy as np
import torch

class TeamAssigner:

    def __init__(self, TEAMS):
        """
        Initialize the TeamAssigner with team colors.
        """
        self.TEAM_COLORS = TEAMS
    '''
    def rgb_to_hsv(self, rgb):
        """
        Convert an RGB color to HSV format.

        Args:
            rgb (tuple): RGB color as (R, G, B).

        Returns:
            tuple: HSV color as (H, S, V).
        """
        rgb_np = np.uint8([[rgb]])
        hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
        return tuple(hsv_np[0][0])

    def calculate_color_coverage(self, image, color_definitions):
        """
        Calculate the percentage of the image covered by the specified colors.

        Args:
            image (np.ndarray): Input image in BGR format.
            color_definitions (list): List of color definitions (RGB + tolerance).

        Returns:
            float: Percentage of image covered by the specified colors.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        for color_def in color_definitions:
            hsv = self.rgb_to_hsv(color_def["rgb"])
            tol = color_def["tolerance"]
            lower_bound = np.array([hsv[0] - tol[0], hsv[1] - tol[1], hsv[2] - tol[2]])
            upper_bound = np.array([hsv[0] + tol[0], hsv[1] + tol[1], hsv[2] + tol[2]])
            
            current_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            mask = cv2.bitwise_or(mask, current_mask)

        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        covered_pixels = np.sum(mask > 0)
        return (covered_pixels / total_pixels) * 100

    def preprocess_image(self, image):
        """
        Ensure the input image is in NumPy array format and BGR color space.

        Args:
            image (torch.Tensor | np.ndarray): Input image.

        Returns:
            np.ndarray: Image in BGR format.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image passed to preprocess_image.")

        if isinstance(image, torch.Tensor):
            # Move to CPU, permute dimensions, and convert to NumPy
            image = image.permute(1, 2, 0).cpu().numpy()
            # Ensure the values are in [0, 255] and type is uint8
            if image.max() <= 1.0:  # Normalize if image is [0, 1]
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure it's in BGR format (OpenCV default)
        if image.shape[-1] == 3:  # Already (H, W, C)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image


    def classify_team(self, image):
        """
        Classify the image to a team based on color presence.

        Args:
            image (torch.Tensor | np.ndarray): Input image in BGR or RGB format.

        Returns:
            tuple: (Team name, Dominant BGR color of the team).
        """
        # Preprocess to ensure correct format
        image = self.preprocess_image(image)

        percentage_from_top = 0.2
        percentage_from_bottom = 0.2
        percentage_from_left = 0.2
        percentage_from_right = 0.2

        crop_top = int(image.shape[0] * percentage_from_top)
        crop_bottom = int(image.shape[0] * (1 - percentage_from_bottom))
        crop_left = int(image.shape[1] * percentage_from_left)
        crop_right = int(image.shape[1] * (1 - percentage_from_right))

        image = image[crop_top:crop_bottom, crop_left:crop_right]

        team_scores = {}
        for team, colors in self.TEAM_COLORS.items():
            coverage = self.calculate_color_coverage(image, colors)
            team_scores[team] = coverage

        # Determine the team with the highest coverage
        assigned_team = max(team_scores, key=team_scores.get)

        # Retrieve the first RGB color for the assigned team and convert to BGR
        dominant_rgb = self.TEAM_COLORS[assigned_team][0]["rgb"]
        dominant_bgr = dominant_rgb[::-1]  # Reverse RGB to BGR

        return assigned_team, dominant_bgr
        '''
    
    ########################################## assign a team based on the yolo object classification model result
    def assign_team(self, result):
        
        assigned_team = 'Unassigned'
        assigned_color = (255, 255, 255)

        if isinstance(result, list):
            # Handle batched results: take the first result
            result = result[0]

        # Access class names from the model
        class_names = result.names if hasattr(result, 'names') else None
        
        if not class_names:
            print("Class names are not available in the result.")
            return

        if result.probs is not None:
            # Convert probabilities to a Python dictionary
            probs_dict = {class_names[i]: float(prob) for i, prob in enumerate(result.probs.data)}
            
            # Get the most probable class and confidence
            top_class = max(probs_dict, key=probs_dict.get)
            confidence = probs_dict[top_class]
            
            #print(f"Top Classification: {top_class}, Confidence: {confidence:.2f}")
            assigned_team = top_class

            # Optional: Print all probabilities
            #print("All Probabilities:", probs_dict)

        else:
            print("No classification probabilities available.")

        # get the matching color from the teams 
        if assigned_team == 'Aphrodite Wanderers':
            assigned_color = (0,0,0)
        if assigned_team == 'Polis':
            assigned_color = (40, 200, 75)
        if assigned_team == 'Akamas':
            assigned_color = (190, 195, 190)
        if assigned_team == 'West Coast':
            assigned_color = (105, 120, 180)
        if assigned_team == 'Goalkeeper':
            assigned_color = (232,224,132)
        if assigned_team == 'Referee':
            assigned_color = (96, 136, 250)
        if assigned_team == 'Unassigned':
            assigned_color = (255, 255, 255)
            
        return (assigned_team,assigned_color)