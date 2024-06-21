# To organize this script into a module structure, we can divide it into multiple files based on functionality. Here is an appropriate module structure

## Module Structure

```shell
cataract_annotator/
│
├── __init__.py
├── main.py
├── gui.py
├── video_processing.py
├── audio_processing.py
└── utils.py
```

### Explanation of Each File

1. **`__init__.py`**:
   - This file makes the folder a package.

2. **`main.py`**:
   - Entry point of the application, initializing the GUI and starting the main loop.

3. **`gui.py`**:
   - Functions and logic related to the GUI using Dear PyGui.

4. **`video_processing.py`**:
   - Functions related to video loading, playback, and annotation.

5. **`audio_processing.py`****:
   - Functions related to audio recording and saving.

6. **`utils.py`**:
   - Utility functions and any additional helpers.

### Code for Each File

**`main.py`**:

```py
from cataract_annotator.gui import start_gui

if __name__ == "__main__":
    start_gui()
```

**`gui.py`**:

```py
import dearpygui.dearpygui as dpg
from cataract_annotator.video_processing import setup_video_player, draw_annotation, play_video, pause_video, video_loop
from cataract_annotator.audio_processing import audio_callback

# Global variables
user_name = ""
user_level = ""

def start_callback(sender, app_data):
    global user_name, user_level
    user_name = dpg.get_value("Name")
    user_level = dpg.get_value("Level")
    dpg.configure_item("start_window", show=False)
    dpg.configure_item("file_dialog_id", show=True)

def file_callback(sender, app_data):
    global video_path
    video_path = app_data['file_path_name']
    dpg.configure_item("file_dialog_id", show=False)
    setup_video_player(video_path)

def start_gui():
    dpg.create_context()

    # Create starting page
    with dpg.window(label="Start", id="start_window"):
        dpg.add_text("Enter your details to start annotating the video:")
        dpg.add_input_text(label="Name", tag="Name")
        dpg.add_combo(label="Level", items=["Beginner", "Intermediate", "Advanced"], tag="Level")
        dpg.add_button(label="Start", callback=start_callback)

    # Create file dialog
    with dpg.file_dialog(directory_selector=False, show=False, callback=file_callback, id="file_dialog_id"):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".mp4", color=(150, 255, 150, 255))

    dpg.create_viewport(title='Cataract-1K Video Annotation Tool', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback):
        while dpg.is_dearpygui_running():
            video_loop()

    dpg.start_dearpygui()
    dpg.destroy_context()
```

**`video_processing.py`**:

```py
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

drawing = False
last_point = None
frames = []
play = False
cap = None
fps = None
frame_size = None
frame_delay = None

def draw_annotation(sender, app_data):
    global last_point, drawing, last_frame, play
    if not play:
        x, y = dpg.get_mouse_pos()
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            if not drawing:
                drawing = True
                last_point = (x, y)
            else:
                cv2.line(last_frame, last_point, (x, y), (0, 255, 0), 1)
                frames.append(last_frame.copy())
                last_point = (x, y)
                dpg.set_value("video_texture", cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        else:
            if drawing:
                drawing = False
                cv2.line(last_frame, last_point, (x, y), (0, 255, 0), 1)
                frames.append(last_frame.copy())
                dpg.set_value("video_texture", cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))

def play_video():
    global play
    play = True

def pause_video():
    global play
    play = False

def video_loop():
    global last_frame, play, cap, frame_delay
    if cap.isOpened():
        if play:
            ret, frame = cap.read()
            if ret:
                last_frame = frame.copy()
                frames.append(last_frame)
                dpg.set_value("video_texture", cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        dpg.render_dearpygui_frame()

def setup_video_player(video_path):
    global cap, fps, frame_size, frame_delay, play
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    frame_delay = int((1/fps)*1000)
    play = False

    dpg.add_texture_registry(label="texture_registry")
    dpg.add_dynamic_texture(width=frame_size[0], height=frame_size[1], default_value=cv2.cvtColor(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8), cv2.COLOR_BGR2RGB), tag="video_texture", parent="texture_registry")

    with dpg.window(label="Video Player", width=frame_size[0], height=frame_size[1] + 60):
        dpg.add_image("video_texture")
        dpg.add_button(label="Play", callback=play_video)
        dpg.add_same_line()
        dpg.add_button(label="Pause", callback=pause_video)
```

**`audio_processing.py`**:

```py
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd

audio_data = []
samplerate = 44100
channels = 2

def audio_callback(indata, frames, time, status):
    audio_data.append(indata.copy())

def save_audio(output_audio_path):
    audio_array = np.concatenate(audio_data, axis=0)
    write(output_audio_path, samplerate, audio_array)
```

**`utils.py`**:

```py
import subprocess

def combine_audio_video(output_video_path, output_audio_path, audio_delay=-1):
    command = [
        'ffmpeg',
        '-i', output_video_path,
        '-itsoffset', str(audio_delay),
        '-i', output_audio_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-shortest',
        output_video_path.replace('.mp4', '_annotated.mp4')
    ]
    subprocess.run(command)
```

### How to Run

1. **Install the package**: Create a `setup.py` if you want to make this an installable package.
2. **Run the main script**: Execute `main.py` to start the application.

This modular structure enhances maintainability, readability, and scalability. Each component of the application is neatly separated into its respective module, allowing for easier development and debugging.
