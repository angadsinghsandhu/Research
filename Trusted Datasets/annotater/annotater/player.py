# General Imports
import cv2, os, threading, queue, json, time
from tqdm import tqdm
from tkinter import messagebox
import sounddevice as sd, time as t
from ffpyplayer.player import MediaPlayer

# Local Imports
from data import Data
from config import config
from annotater.controller import ControlWindow


class VideoPlayer:
    def __init__(self, app, file_name, done_event):
        
        # Video Player Objects
        self.app = app
        self.cap, self.frame, self.ret = None, None, None
        self.done_event = done_event  # Event to signal completion

        # Variables
        self.file_path, self.file_name = config.in_path, file_name
        self.cwd, self.out_path = os.getcwd(), config.out_path
        self._data = None
        self.last_frame, self.last_frame_idx, self.last_point = None, None, None

        # FLags
        self.paused, self.drawing = False, False
        self.start_counter, self.pause_frame = None, None

        # widgets
        self.control_window, self.control_frame = None, None
        self.play_pause_button, self.seeker = None, None

        # Queue for thread-safe communication
        self.command_queue = queue.Queue()
        
        # Lock for thread-safe data updates
        self.lock = threading.Lock()

        # Open the video player
        if not os.path.exists(f"{self.file_path}\{self.file_name}"):
            # check if file exists
            messagebox.showerror("Error", "File not found.")
            return
        else:
            self.cap = cv2.VideoCapture(f"{self.file_path}\{self.file_name}")
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file.")
                return
            
        # get last frame and its index
        self.last_frame_idx = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.last_frame_idx)
        _, self.last_frame = self.cap.read()
        self.last_frame_idx, self.curr_frame_idx = None, None
        
        # reset cap to the first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Audio variables
        self.samplerate = 44100
        self.channels = 2

        self._data = Data(
            in_path=self.file_path, out_path=self.out_path,
            name=self.file_name, 
            frame_width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            frame_height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
            fps=int(self.cap.get(cv2.CAP_PROP_FPS)), 
            fc=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            sample_rate=self.samplerate, channels=self.channels)
        
        self.frame_delay = int((1 / self._data.frame_rate) * 1000)

        # set up audio stream
        # self.audio_stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=self.audio_callback)

        # set up video player
        self.video_thread = threading.Thread(target=self.main_loop, daemon=True)

        # Ensure control window is only set up once
        self.control_window = ControlWindow(self.app, self.file_name, self)

        # Start the video player thread
        self.video_thread.start()

        # Start the control window
        self.control_window.mainloop()

    def audio_callback(self, indata, frames, time, status):
        if self.start_counter is None:
            self.start_counter = t.perf_counter()
        timestamp = t.perf_counter() - self.start_counter
        self._data.add_audio_data(timestamp, indata.copy())

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.drawing:
            self.drawing = True
            self._data.add_annotation("start", (x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._data.add_annotation("move", (x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self._data.add_annotation("end", (x, y))

    def draw_annotations(self):
        """Draw annotations on the given frame based on the frame index."""
        _annotation = self._data.get_last_annotation()
        largest_key = max(self._data.annotations.keys())
        if _annotation is not None and abs(largest_key - self._data.get_frames_length) <= 5:
            command, (x, y) = _annotation
            if command == "start":
                self.start_point = (x, y)
            elif command == "move":
                cv2.line(self.frame, self.start_point, (x, y), (0, 0, 255), 3)
                self.start_point = (x, y)
            elif command == "end":
                cv2.line(self.frame, self.start_point, (x, y), (0, 0, 255), 3)
                self.start_point = None
                self.drawing = False

    def main_loop(self):
        cv2.namedWindow("Video Player")
        cv2.setMouseCallback("Video Player", self.mouse_callback)

        total_frames = self._data.get_max_frames
        self.start_counter = t.perf_counter()  # Start high-resolution timer

        # start audio stream
        with sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=self.audio_callback):
            with tqdm(total=total_frames, desc="Processing Frames") as self.pbar:
                while self.cap.isOpened():
                    start_time = t.time()
                    current_counter = t.perf_counter() - self.start_counter

                    self.curr_frame_idx = curr_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if curr_frame >= total_frames: curr_frame = total_frames - 1
                    
                    # update seeker
                    self.control_window.seek_var.set(curr_frame)

                    # Check for commands
                    try:
                        command = self.command_queue.get_nowait()

                        if command == 'pause': self.toggle_pause()
                        elif isinstance(command, tuple) and command[0] == 'seek': self.seek(command[1])
                        elif command == 'restart': self.restart()

                    except queue.Empty:
                        pass
                    
                    # Check for pause
                    if not self.paused:
                        self.ret, self.frame = self.cap.read()
                        if not self.ret: 
                            self.pbar.total += 1
                            self.frame = self.last_frame.copy()
                        self.pause_frame = self.frame.copy()
                    else:
                        self.frame = self.pause_frame.copy()
                        self.pbar.total += 1
                        self._data.increment_max_frame()

                    if self.drawing: self.draw_annotations()

                    # Display the frame
                    cv2.imshow("Video Player", self.frame)

                    # add current frame to data
                    self._data.add_curr_frame(current_counter, curr_frame)

                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    # print(f"Key: {key}")

                    # Update progress bar
                    self.pbar.update(1)

                    # Update frame delay
                    elapsed_time = t.time() - start_time
                    remaining_time = max(0, self.frame_delay / 1000 - elapsed_time)  # Convert frame_delay to seconds
                    if remaining_time > 0:
                        t.sleep(remaining_time)

    def toggle_pause(self):
        self.paused = not self.paused

    def seek(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._data.update_curr_frame(frame_number)
        self.curr_frame_idx = frame_number

    def restart(self):
        # Reset the video player
        self.paused, self.drawing = False, False

        # reset cap to the first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # clean data
        self._data.clean()
        self._data.update_max_frames(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._data.update(
            in_path=self.file_path, out_path=self.out_path,
            name=self.file_name, 
            frame_width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            frame_height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
            fps=int(self.cap.get(cv2.CAP_PROP_FPS)), 
            fc=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            sample_rate=self.samplerate, channels=self.channels)
        
        # clear command queue
        while not self.command_queue.empty():
            _ = self.command_queue.get()

        # Reset tqdm progress bar
        self.pbar.reset(total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.pbar.refresh()

    def close(self, save=False):
        # relese video capture and stop audio stream
        self.cap.release()
        sd.stop()

        # save data if required
        if save: self._data.save_data(self.app)

        # # clean up
        # self._data = None

        # close windows and threads
        cv2.destroyAllWindows()
        self.done_event.set()
        self.video_thread.join(1)
        self.control_window.quit()
        self.control_window.destroy()
        self.app.deiconify()

class AnnotatedPlayer:
    def __init__(self, watch_file, meta_file):
        self.watch_file = watch_file
        self.meta_file = meta_file

        if not os.path.exists(watch_file):
            messagebox.showerror("Error", "File not found.")
            return
        else:
            self.cap = cv2.VideoCapture(watch_file)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file.")
                return

        if not os.path.exists(meta_file):
            messagebox.showerror("Error", "File metadata not found.")
            return
        else:
            self.meta = json.load(open(meta_file, "r"))

        self.player  = MediaPlayer(watch_file)
        self.audio_on = True

        self.start_time = None
        self.last_point = None

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.meta['metadata']['frame_rate']
        self.frame_delay = int((1 / self.frame_rate) * 1000)

        self.show()

    def show(self):
        # open video player and display annotations
        cv2.namedWindow("Annotater Player")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if self.audio_on: audio_frame, val = self.player.get_frame()
            
            if ret:
                frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"Frame Number: {frame_number}")
                if str(frame_number) in self.meta:
                    action, point = self.meta[str(frame_number)]
                    if action == "start":
                        self.last_point = tuple(point)
                    elif action == "move":
                        cv2.line(frame, self.last_point, tuple(point), (0, 0, 255), 3)
                        self.last_point = tuple(point)
                    elif action == "end":
                        cv2.line(frame, self.last_point, tuple(point), (0, 0, 255), 3)
                        self.last_point = None

                if self.audio_on and val != 'eof' and audio_frame is not None: _, t = audio_frame
                elif val == 'eof': 
                    self.audio_on = False
                    continue

                cv2.imshow("Annotater Player", frame)
                if cv2.waitKey(self.frame_delay) & 0xFF == ord('q'): break
            else: break

        self.close()

    def close(self):
        self.cap.release()
        self.player.close_player()
        cv2.destroyAllWindows()