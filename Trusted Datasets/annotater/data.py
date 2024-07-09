from functools import lru_cache, cached_property
from tkinter import filedialog
import subprocess, cv2
from scipy.io.wavfile import write
import numpy as np
import time, json

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
    def __init__(self, path, name, fps=30.0):
        self.in_path = path     # Path to the video file selected
        self.name = name        # Default name of the file (same name as selected file)
        self.FPS = fps          # Frames per second of the selected video
        self.out_path = None    # Path where the video file will be saved

        self.frames = []
        self.audio_data =  []
        self.annotations = {}

        self.init_time = time.time()
        self.curr_time = None
        self.end_time = None

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
    
    # MOTHODS TO GET DATA LENGTH

    @cached_property
    def get_video_data_length(self):
        return len(self.frames)
    
    @cached_property
    def get_audio_data_length(self):
        return len(self.audio_data)
    
    @cached_property
    def get_annotations_length(self):
        return len(self.annotations)

    # METHODS TO ADD DATA

    def add_video_data(self, frame):
        self.frames.append(frame)

    def add_audio_data(self, audio):
        self.audio_data.append(audio)

    def add_annotation(self, annotation):
        """Adds annotations with timestamp, potentially expensive if called frequently"""
        self.curr_time = time.time()
        time_diff = self.curr_time - self.init_time
        self.annotations[time_diff] = annotation

    # METHODS TO SAVE DATA

    def save_video_data(self):
        """Function to save video and audio, and merge them using FFmpeg"""
        try:
            self.out_path = filedialog.asksaveasfilename(defaultextension=f"{self.name}.mp4", filetypes=[("MP4 files", "*.mp4")])
            if not self.out_path: return

            # Assume all frames are the same size as the first frame
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.out_path, fourcc, self.FPS, (width, height))  # Use actual frame size
            
            for frame in self.frames: out.write(frame)
            out.release()

            # Combine video and audio
            subprocess.run([
                'ffmpeg', 
                '-i', self.out_path, 
                '-c:v', 'copy', 
                '-c:a', 'aac', 
                '-strict',  'experimental', 
                self.out_path.replace('.mp4', '_annotated.mp4')])
        except Exception as e:
            print(f"Error saving video data: {e}")

    def save_audio_video_data(self):
        """Function to save video and audio, and merge them using FFmpeg"""
        try:
            self.out_path = filedialog.asksaveasfilename(defaultextension=f"{self.name}.mp4", filetypes=[("MP4 files", "*.mp4")])
            if not self.out_path: return

            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.out_path, fourcc, self.FPS, (width, height))

            for frame in self.frames: out.write(frame)
            out.release()

            output_audio_path = self.out_path.replace(".mp4", ".wav")
            write(output_audio_path, 44100, self.combined_audio)

            subprocess.run([
                'ffmpeg',
                '-i', self.out_path,
                '-i', output_audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                self.out_path.replace('.mp4', '_annotated.mp4')
            ])

        except Exception as e:
            print(f"Error saving audio-video data: {e}")

    def save_annotations(self):
        """Save annotations to a JSON file"""
        try:
            with open(f"{self.in_path}\\{self.name}_annotations.json", "w") as file:
                json.dump(self.annotations, file)
        except Exception as e:
            print(f"Error saving annotations: {e}")
