# General Imports
import subprocess, cv2, time, json
import numpy as np
from functools import lru_cache, cached_property
from customtkinter import filedialog
from tkinter import messagebox
from scipy.io.wavfile import write

class Data:
    """
    Class to store video and audio data, and annotations.

    Parameters:
        - path (str): Path to save video and audio files.
        - name (str): Name of the video file.
        - fps (float): Frames per second of the video.

    Methods:
        - add_video_data(frame): Add video data to the frames list.
        - add_audio_data(audio): Add audio data to the audio list.
        - add_annotation(annotation): Add annotation to the annotations dictionary.
        - save_video_data(): Save video data to a file.
        - save_audio_video_data(): Save audio and video data to a file.
        - save_annotations(): Save annotations to a file.
    """
    def __init__(self, in_path, out_path, name, frame_width, frame_height, fps=30.0, fc=10225):

        def print_data_object_info(name, in_path, out_path, fps, fc):
            table_data = [
                ["Description", "Value"],
                ["Data Object Created for", name],
                ["Input Path", in_path],
                ["Output Path", out_path],
                ["Name", name],
                ["FPS", fps],
                ["Frame Count", fc]
            ]

            column_width = max(len(str(item)) for row in table_data for item in row) + 2
            separator = "+" + "-" * (column_width * 2 + 1) + "+"

            print(separator)
            for row in table_data:
                print("|" + "|".join(str(item).center(column_width) for item in row) + "|")
                print(separator)

        print_data_object_info(name, in_path, out_path, fps, fc)

        self.in_path = in_path      # Path to the video file selected
        self.out_path = out_path    # Path where the video file will be saved
        self.name = name            # Default name of the file (same name as selected file)
        
        self.FPS = fps                      # Frames per second of the selected video
        self.frame_count = int(fc)          # Frame count of the selected video
        self.frame_width = frame_width      # Width of the frame
        self.frame_height = frame_height    # Height of the frame

        self.frames = []
        self.audio_data =  []
        self.annotations = {}

        self.curr_frame = 0
        self.max_frame = int(fc)

    # METHODS TO COMBINE DATA

    @cached_property
    def combined_audio(self):
        """Combines all audio data into a single array, expensive operation hence cached"""
        return np.concatenate(self.audio_data, axis=0)

    # METHODS TO GET DATA

    @lru_cache(maxsize=128)
    def get_video_frame(self, index):
        return self.frames[index]
    
    @lru_cache(maxsize=128)
    def get_audio_data(self, index):
        return self.audio_data[index]
    
    @lru_cache(maxsize=128)
    def get_annotation(self, timestamp):
        return self.annotations[timestamp]
    
    @cached_property
    def get_curr_frame(self):
        return self.curr_frame
    
    @cached_property
    def get_max_frame(self):
        return self.max_frame
    
    # FRAME HANDLING

    @lru_cache(maxsize=128)
    def get_current_frame(self):
        return self.frames[self.curr_frame]
    
    # MOTHODS TO GET DATA LENGTH

    @cached_property
    def get_frames_length(self):
        return len(self.frames)
    
    @cached_property
    def get_audio_data_length(self):
        return len(self.audio_data)
    
    @cached_property
    def get_annotations_length(self):
        return len(self.annotations)

    # METHODS TO ADD DATA

    def add_curr_frame(self, frame):
        if frame >= self.frame_count:
            print(f"Frame index {frame} is out of range.")
        self.frames.append(frame)

    def add_audio_data(self, audio):
        self.audio_data.append(audio)

    def add_annotation(self, annotation):
        """Adds annotations with timestamp, potentially expensive if called frequently"""
        self.curr_time = time.time()
        time_diff = self.curr_time - self.init_time
        self.annotations[time_diff] = annotation

    # METHODS TO UPDATE DATA

    def update(self, in_path, out_path, name, frame_width, frame_height, fps, fc):
        self.in_path = in_path
        self.out_path = out_path
        self.name = name
        self.FPS = fps
        self.frame_count = fc
        self.frame_width = frame_width
        self.frame_height = frame_height

    def update_curr_frame(self, frame):
        self.curr_frame = frame

    def update_max_frame(self, frame):
        self.max_frame = frame

    def increment_max_frame(self):
        self.max_frame += 1

    # METHODS TO SAVE DATA

    def process_video_data(self):
        # open the video file
        cap = cv2.VideoCapture(f"{self.in_path}\{self.name}")

        # check if the video file is opened
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return
        
        # read the video file frame by frame
        vid, fr_cnt = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vid.append(frame)
            fr_cnt += 1

        # Get original video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Assert that the original properties match the given values
        assert original_fps == self.FPS, f"FPS mismatch: {original_fps} != {self.FPS}"
        assert original_width == self.frame_width, f"Frame width mismatch: {original_width} != {self.frame_width}"
        assert original_height == self.frame_height, f"Frame height mismatch: {original_height} != {self.frame_height}"
        assert fr_cnt == self.frame_count, "Frame count mismatch"

        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{self.out_path}\{self.name}", fourcc, self.FPS, (self.frame_width, self.frame_height))

        # Write frames to the output video in the specified order
        print(f"Writing {len(self.frames)} frames to {self.out_path}\{self.name}")
        for index in self.frames:
            if index < len(vid):
                out.write(vid[index])
            else:
                print(f"Warning: Frame index {index} is out of range.")
        print(f"Video saved to {self.out_path}\{self.name}")


    def save_data(self):
        """Function to save video and audio, and merge them using FFmpeg"""
        try:
            if not self.out_path:
                self.out_path = filedialog.askdirectory(filetypes=[("MP4 files", "*.mp4")])

            # Check if there's audio data and video data synchronization is needed
            if len(self.audio_data) > 0:
                assert len(self.frames) == len(self.audio_data), "Audio and video data are not in sync"

            # Save video data
            self.process_video_data()

            # Check if audio data and annotations are present
            if len(self.audio_data) > 0:
                audio_name = self.name.replace(".mp4", ".wav")
                write(f"{self.out_path}\{audio_name}", 44100, self.combined_audio)

                subprocess.run([
                    'ffmpeg',
                    '-i', f"{self.out_path}\{self.name}",
                    '-i', f"{self.out_path}\{audio_name}",
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    self.out_path.replace('.mp4', '_annotated.mp4')
                ])
            
            if self.annotations:
                """Save annotations to a JSON file"""
                try:
                    with open(f"{self.out_path}\\{self.name}_annotations.json", "w") as file:
                        json.dump(self.annotations, file)
                except Exception as e:
                    print(f"Error saving annotations: {e}")

        except Exception as e:
            print(f"Error saving audio-video-annotation data: {e}")

    # METHODS TO DELETE DATA

    def clean(self):
        """Function to clean the data object"""
        self.frames = []
        self.audio_data = []
        self.annotations = {}
        self.curr_frame = 0
        self.max_frame = 0