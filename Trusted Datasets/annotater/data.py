# General Imports
import subprocess, cv2, json, os, logging
import numpy as np
from functools import lru_cache, cached_property
from customtkinter import filedialog
from tkinter import messagebox
from scipy.io.wavfile import write

# Custom Imports
from screens import SaveProgress

# Set up logging
logger = logging.getLogger('app')

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
    def __init__(self, in_path, out_path, name, frame_width, frame_height, fps=30, fc=10225, sample_rate=44100, channels=2):

        self.print_data_object_info(name, in_path, out_path, fps, fc, frame_width, frame_height, sample_rate, channels, log=True)

        self.in_path = in_path      # Path to the video file selected
        self.out_path = out_path    # Path where the video file will be saved
        self.name = name            # Default name of the file (same name as selected file)

        self.in_file_path = os.path.join(in_path, name)    # Full path to the selected video file
        self.out_file_path = os.path.join(out_path, name)  # Full path to the selected video file
        
        self.frame_rate = fps               # Frames per second of the selected video
        self.frame_count = fc               # Frame count of the selected video
        self.frame_width = frame_width      # Width of the frame
        self.frame_height = frame_height    # Height of the frame

        self.frames = []                    # List of (timestamp, frame) tuples
        self.audio_data =  []               # List of (timestamp, audio_chunk) tuples
        self.annotations = {}               # Dictionary of annotations with frame index as key

        self.curr_frame = 0
        self.max_frames = fc

        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_name = self.name.replace(".mp4", ".wav")
        self.audio_path = os.path.join(self.out_path, self.audio_name)

        logger.info(f"Data object for {name} initialized")

    # METHODS TO PRINT DATA

    def print_data_object_info(self, name, in_path, out_path, fps, fc, frame_width, frame_height, sample_rate, channels, log=False):
        table_data = [
            ["Description", "Value"],
            ["Data Object Created for", name],
            ["Input Path", in_path],
            ["Output Path", out_path],
            ["Name", name],
            ["FPS", fps],
            ["Frame Count", fc],
            ["Frame Width", frame_width],
            ["Frame Height", frame_height],
            ["Sample Rate", sample_rate],
            ["Channels", channels]
        ]

        column_width = max(len(str(item)) for row in table_data for item in row) + 2
        separator = "+" + "-" * (column_width * 2 + 1) + "+"
        
        if log:
            _str = "| "
            for row in range(1, len(table_data)): _str +=  str(table_data[row][0]) + " ==> " + str(table_data[row][1]) + " | "
            logger.debug(_str.rstrip())
        else:
            print(separator)
            for row in table_data:
                print("|" + "|".join(str(item).center(column_width) for item in row) + "|")
                print(separator)

    # METHODS TO GET DATA

    @lru_cache(maxsize=128)
    def get_video_frame(self, index):
        return self.frames[index]
    
    @lru_cache(maxsize=128)
    def get_audio_data(self, index):
        return self.audio_data[index]
    
    def get_annotation(self, frame_idx):
        if frame_idx not in self.annotations: return None
        return self.annotations[frame_idx]
    
    @cached_property
    def get_curr_frame(self):
        return self.curr_frame
    
    def get_last_annotation(self):
        if self.get_frames_length == 0 or self.get_annotations_length == 0:
            return None
        return self.get_annotation(max(self.annotations.keys()))
    
    @property
    def get_max_frames(self):
        return self.max_frames
    
    # FRAME HANDLING

    @lru_cache(maxsize=128)
    def get_current_frame(self):
        return self.frames[self.curr_frame]
    
    # MOTHODS TO GET DATA LENGTH

    @property
    def get_frames_length(self):
        return len(self.frames)
    
    @property
    def get_audio_data_length(self):
        return len(self.audio_data)
    
    @property
    def get_annotations_length(self):
        return len(self.annotations)

    # METHODS TO ADD DATA

    def add_curr_frame(self, timestamp, frame):
        if frame >= self.frame_count:
            logger.warning(f"Frame index {frame} is out of range.")
        self.frames.append((timestamp, frame))

    def add_audio_data(self, timestamp, audio):
        self.audio_data.append((timestamp, audio))

    def add_annotation(self, command, annotation):
        """Adds annotations with timestamp, potentially expensive if called frequently"""
        if self.get_frames_length == 0:
            logger.error("No frames to annotate.")
            return

        frame_idx = self.get_frames_length - 1
        if frame_idx in self.annotations and self.annotations[frame_idx][0] == "start":
            self.annotations[frame_idx] = ["start", list(annotation)]
        else: self.annotations[frame_idx] = [command, list(annotation)]

        logger.debug(f"Added annotation at frame {frame_idx} with command {command}")

    # METHODS TO UPDATE DATA

    def update(self, in_path, out_path, name, frame_width, frame_height, fps, fc, sample_rate, channels):
        self.in_path = in_path
        self.out_path = out_path
        self.name = name

        self.in_file_path = os.path.join(in_path, name)
        self.out_file_path = os.path.join(out_path, name)

        self.frame_rate = fps
        self.frame_count = fc
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.sample_rate = sample_rate
        self.channels = channels

        logger.info(f"Data() updated for {name}: in_path={in_path}, out_path={out_path}, frame_width={frame_width}, frame_height={frame_height}")

    def update_curr_frame(self, frame):
        self.curr_frame = frame

    def update_max_frames(self, frame):
        self.max_frames = frame

    def increment_max_frame(self):
        self.max_frames += 1

    # METHODS TO COMBINE DATA

    def identify_video_gaps(self, threshold=0.1):
        """Identify gaps in video frames where the time difference exceeds the threshold."""
        video_gaps = []
        for i in range(1, len(self.frames)):
            previous_time, _ = self.frames[i - 1]
            current_time, _ = self.frames[i]
            if current_time - previous_time > threshold:
                video_gaps.append((previous_time, current_time))
        return video_gaps

    def combined_audio(self):
        """Combines all audio data into a single array, removing audio during video gaps."""
        video_gaps = self.identify_video_gaps()
        
        synced_audio = []
        current_audio_index = 0

        for gap_start, gap_end in video_gaps:
            # Add audio before the gap
            while current_audio_index < len(self.audio_data) and self.audio_data[current_audio_index][0] < gap_start:
                synced_audio.append(self.audio_data[current_audio_index][1])
                current_audio_index += 1
            
            # Skip audio during the gap
            while current_audio_index < len(self.audio_data) and self.audio_data[current_audio_index][0] < gap_end:
                current_audio_index += 1

        # Add remaining audio after the last gap
        while current_audio_index < len(self.audio_data):
            synced_audio.append(self.audio_data[current_audio_index][1])
            current_audio_index += 1

        if synced_audio:
            combined_audio = np.concatenate(synced_audio, axis=0)
            logger.debug("Combined audio data successfully")
            return combined_audio
        else:
            logger.error("Synced audio is empty or invalid.")
            return np.array([])

    # METHODS TO SAVE DATA

    def process_video_data(self, progress=None):
        if not self.frames:
            messagebox.showerror("Error", "No frames to save.")
            logger.error("No frames to save")
            return

        # open the video file
        cap = cv2.VideoCapture(self.in_file_path)

        # check if the video file is opened
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            logger.error("Failed to open video file")
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
        assert original_fps == self.frame_rate, f"FPS mismatch: {original_fps} != {self.frame_rate}"
        assert original_width == self.frame_width, f"Frame width mismatch: {original_width} != {self.frame_width}"
        assert original_height == self.frame_height, f"Frame height mismatch: {original_height} != {self.frame_height}"
        assert fr_cnt == self.frame_count, "Frame count mismatch"

        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.out_file_path, fourcc, self.frame_rate, (self.frame_width, self.frame_height))

        # Write frames to the output video in the specified order
        last_index = len(self.frames) - 1
        for timestamp, index in self.frames:
            if index < len(vid):
                out.write(vid[index])
                if progress: progress.update_video_progress(index / last_index)
            else:
                logger.warning(f"Frame index {index} is out of range.")

        logger.info(f"Video data processed and saved to {self.out_file_path}")

    def process_audio_data(self, progress=None):
        # Check if audio data and annotations are present
        if len(self.audio_data) > 0:
            write(self.audio_path, self.sample_rate, self.combined_audio())
            logger.info(f"Audio data saved to {self.audio_path}")
            if progress: progress.update_audio_progress(1.0)
        else: logger.warning("No audio data to save")

    def save_av_and_clean(self, progress=None):
        """Merge audio and video data using FFmpeg"""
        subprocess.run([
            'ffmpeg',
            '-hide_banner',
            '-y',
            '-i', self.out_file_path,
            '-i', self.audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            self.out_file_path.replace('.mp4', '_annotated.mp4')
        ])

        if progress: progress.update_av_progress(1.0)
        logger.info(f"Audio and video merged and saved to {self.out_file_path.replace('.mp4', '_annotated.mp4')}")

        # delete the audio and video data at self.out_file_path and self.audio_path
        if os.path.exists(self.out_file_path):
            os.remove(self.out_file_path)
            logger.debug(f"Deleted temporary video file {self.out_file_path}")
        if os.path.exists(self.audio_path):
            os.remove(self.audio_path) 
            logger.debug(f"Deleted temporary audio file {self.audio_path}")

    def save_annotations(self, progress=None):
        # add metadata to the annotations file
        self.annotations["metadata"] = {
            "video_name": self.name,
            "frame_rate": self.frame_rate,
            "frame_count": self.max_frames,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "sample_rate": self.sample_rate,
            "channels": self.channels
        }

        """Save annotations to a JSON file"""
        try:
            with open(self.out_file_path.replace('.mp4', '_annotated.json'), "w") as file:
                json.dump(self.annotations, file)

            if progress: progress.update_json_progress(1.0)
            logger.info(f"Annotations saved to {self.out_file_path.replace('.mp4', '_annotated.json')}")
        except Exception as e: logger.error(f"Error saving annotations: {e}")

    def save_data(self, app):
        """Function to save video and audio, and merge them using FFmpeg"""
        if not self.out_path:
            self.out_path = filedialog.askdirectory(filetypes=[("MP4 files", "*.mp4")])

        logger.info(f"Saving data for {self.name} to {self.out_path}")

        # Create progress window
        progress = SaveProgress(app, self.name)
        progress.update()

        # Save video data
        self.process_video_data(progress=progress)

        # Save audio data
        self.process_audio_data(progress=progress)

        # Cleanup
        self.save_av_and_clean(progress=progress)

        # Save annotations
        self.save_annotations(progress=progress)

        # update progress window title
        progress.update_title_on_save()
        logger.info(f"Data for {self.name} saved successfully")

    # METHODS TO DELETE DATA

    def clean(self):
        """Function to clean the data object"""
        self.frames = []
        self.audio_data = []
        self.annotations = {}
        self.curr_frame = 0
        self.max_frame = 0
        logger.debug("Data object cleaned")