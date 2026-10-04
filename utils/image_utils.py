import cv2
import numpy as np
import torch

class ImageUtils:

    @staticmethod
    def create_tensor_and_resize(frame,
                                 MODEL_WIDTH,
                                 MODEL_HEIGHT,
                                 ORIGINAL_VIDEO_WIDTH,
                                 ORIGINAL_VIDEO_HEIGHT):

        #print(f"Starting shape: {frame.shape}")
        #print(f"Starting range: {frame.min()} to {frame.max()}")
        #print(f"Starting dtype: {frame.dtype}")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img = frame

        # Resize to dimensions divisible by 32
        target_size = (MODEL_WIDTH, MODEL_HEIGHT)
        img = cv2.resize(img, target_size)
        
        #print(f"\nAfter resize shape: {img.shape}")
        #print(f"After resize range: {img.min()} to {img.max()}")

        # Convert to tensor with proper format
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW format
        img_tensor = img_tensor / 255.0  # Normalize to [0-1]
        
        # Don't add batch dimension - this will be handled by the batching process
        # img_tensor = img_tensor.unsqueeze(0)  # Removed this line

        #print(f"\nFinal tensor shape: {img_tensor.shape}")
        #print(f"Final tensor range: {img_tensor.min():.3f} to {img_tensor.max():.3f}")
        #print(f"Final tensor dtype: {img_tensor.dtype}")
        
        # Ensure correct dimensions (CHW format, without batch dimension)
        expected_shape = torch.Size([3, MODEL_HEIGHT, MODEL_WIDTH])
        if img_tensor.shape != expected_shape:
            raise ValueError(f"Incorrect tensor shape {img_tensor.shape}. Expected {expected_shape}")

        return img_tensor

    
    @staticmethod
    def create_team_assigner_tensor(image, target_width, target_height):

        """take a tensor image and resize it to the correct size for the team assigner model"""

        #print("\nDEBUG INPUT:")
        #print(f"Input type: {type(image)}")
        if isinstance(image, torch.Tensor):
            #print(f"Input tensor shape: {image.shape}")
            #print(f"Input tensor range: {image.min():.3f} to {image.max():.3f}")
            #print(f"Input tensor dtype: {image.dtype}")
            
            # Keep the original values but convert to numpy for processing
            img = (image.cpu().numpy() * 255).astype(np.uint8)  # Scale back to 0-255 range
            if img.shape[0] == 3:  # If in CHW format
                img = img.transpose(1, 2, 0)  # Convert to HWC
        else:
            img = image.copy()
            #print(f"Input numpy shape: {img.shape}")
            #print(f"Input numpy range: {img.min()} to {img.max()}")
            #print(f"Input numpy dtype: {img.dtype}")

        #print("\nDEBUG AFTER CONVERSION:")
        #print(f"Working with shape: {img.shape}")
        #print(f"Working with range: {img.min()} to {img.max()}")
        #print(f"Working with dtype: {img.dtype}")

        # Convert color space
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to dimensions divisible by 32
        target_size = (target_width, target_height)  # Should be (192, 192)
        img = cv2.resize(img, target_size)
        
        #print("\nDEBUG AFTER RESIZE:")
        #print(f"After resize shape: {img.shape}")
        #print(f"After resize range: {img.min()} to {img.max()}")
        
        # Convert to tensor with proper format
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)  # HWC to CHW format
        img = img / 255.0  # Normalize to [0-1]
        img = img.unsqueeze(0)  # Add batch dimension
        
        #print("\nDEBUG FINAL TENSOR:")
        #print(f"Final tensor shape: {img.shape}")
        #print(f"Final tensor range: {img.min():.3f} to {img.max():.3f}")
        #print(f"Final tensor dtype: {img.dtype}")
        
        return img