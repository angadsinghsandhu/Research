# modules/video_utils.py
"""
Utility functions for video processing.
"""

import cv2 as cv
import numpy as np
import dearpygui.dearpygui as dpg

vid = None

def initialize_video():
    """
    Initialize the video capture and setup the initial texture.
    """
    global vid
    vid = cv.VideoCapture(0)
    ret, frame = vid.read()
    print_frame_info(vid, frame)

    # Process frame data for Dear PyGui
    texture_data = process_frame_for_texture(frame)

    # Register texture
    with dpg.texture_registry(show=True):
        dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)

def update_frame():
    """
    Update the texture with the latest frame from the video capture.
    """
    if vid is not None:
        ret, frame = vid.read()
        if ret:
            texture_data = process_frame_for_texture(frame)
            dpg.set_value("texture_tag", texture_data)

def cleanup_video():
    """
    Release the video capture.
    """
    if vid is not None:
        vid.release()

def print_frame_info(vid, frame):
    """
    Print information about the video frame.
    """
    frame_width = vid.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = vid.get(cv.CAP_PROP_FRAME_HEIGHT)
    video_fps = vid.get(cv.CAP_PROP_FPS)
    print(f"Frame Width: {frame_width}")
    print(f"Frame Height: {frame_height}")
    print(f"FPS: {video_fps}")

    print("Frame Array:")
    print(f"Array is of type: {type(frame)}")
    print(f"No. of dimensions: {frame.ndim}")
    print(f"Shape of array: {frame.shape}")
    print(f"Size of array: {frame.size}")
    print(f"Array stores elements of type: {frame.dtype}")

def process_frame_for_texture(frame):
    """
    Process the video frame for Dear PyGui texture.
    """
    data = np.flip(frame, 2)  # Convert BGR to RGB
    data = data.ravel()  # Flatten to 1D
    data = np.asfarray(data, dtype='f')  # Convert to 32-bit floats
    texture_data = np.true_divide(data, 255.0)  # Normalize to [0, 1]
    
    return texture_data
