import sys
print(f"Python version: {sys.version}")

import moviepy
print(f"MoviePy version: {moviepy.__version__}")

try:
    from moviepy.editor import VideoFileClip
    print("Successfully imported VideoFileClip")
except ImportError as e:
    print(f"Error: {e}")
    print(f"MoviePy path: {moviepy.__file__}")
    print("\nContents of moviepy directory:")
    import os
    moviepy_dir = os.path.dirname(moviepy.__file__)
    print(os.listdir(moviepy_dir))