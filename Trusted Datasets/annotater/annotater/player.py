import cv2, os, time, threading, queue
from tqdm import tqdm
from PIL import Image
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import messagebox, Tk, DoubleVar
from data import Data
from config import config

# TODO : add audio callback function
# TODO: setup audio input
# TODO : implement draw_annotation later on in seperate dictionary
# TODO : Make seeker faster and in real time

class VideoPlayer:
    def __init__(self, app, file_name, done_event):
        
        # Video Player Objects
        self.app = app
        self.cap = None
        self.done_event = done_event  # Event to signal completion

        # Variables
        self.file_path, self.file_name = config.in_path, file_name
        self.cwd, self.out_path = os.getcwd(), config.out_path
        self._data = None
        self.last_frame, self.last_point = None, None

        # FLags
        self.paused, self.drawing = False, False

        # widgets
        self.control_window, self.control_frame = None, None
        self.play_pause_button, self.seeker = None, None
        self.seek_var = DoubleVar()

        # Queue for thread-safe communication
        self.command_queue = queue.Queue()
        
        # Lock for thread-safe data updates
        self.lock = threading.Lock()

        # Open the video player
        # check if file exists
        if not os.path.exists(f"{self.file_path}\{self.file_name}"):
            messagebox.showerror("Error", "File not found.")
            return
        else:
            self.cap = cv2.VideoCapture(f"{self.file_path}\{self.file_name}")
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file.")
                return

        self._data = Data(
            in_path=self.file_path, out_path=self.out_path, 
            name=self.file_name, 
            frame_width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            frame_height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
            fps=self.cap.get(cv2.CAP_PROP_FPS), fc=self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.frame_delay = int((1 / self._data.FPS) * 1000)

        # set up video player
        self.video_thread = threading.Thread(target=self.main_loop, daemon=True)

        # Ensure control window is only set up once
        if self.control_window is None:
            self.control_window = ControlWindow(self.app, self.file_name, self)

    # def on_mouse_click(self, event):
    #     if event.num == 1 and self.paused:  # Ensure this happens only if the video is paused
    #         self.drawing = True
    #         self.last_point = (event.x, event.y)
    #     elif not self.paused:
    #         self.drawing = False

    # def on_mouse_move(self, event):
    #     if self.drawing and self.paused:
    #         x, y = event.x, event.y
    #         self._data.add_annotation(((x, y), self.last_point))
    #         self.last_point = (x, y)

    def mouse_callback(self, event, x, y, flags, param):
        pass
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     self.paused = not self.paused

    def main_loop(self):
        cv2.namedWindow("Video Player")
        cv2.setMouseCallback("Video Player", self.mouse_callback)

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Processing Frames") as pbar:
            while self.cap.isOpened():
                start_time = time.time()
                curr_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                try:
                    command = self.command_queue.get_nowait()
                    if command == 'pause': self.toggle_pause()
                    elif isinstance(command, tuple) and command[0] == 'seek': self.seek_to_frame(command[1])
                except queue.Empty:
                    pass

                if not self.paused:
                    ret, frame = self.cap.read()
                    
                    # TODO : think about ending
                    if not ret: break
                        
                    self.last_frame = frame.copy()
                    self._data.add_curr_frame(curr_frame)
                    cv2.imshow("Video Player", frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    self._data.add_curr_frame(curr_frame)
                    key = cv2.waitKey(1) & 0xFF  # Ensure responsiveness during pause
                    if key == ord('q'): break

                # Update progress bar
                pbar.update(1)

                elapsed_time = time.time() - start_time
                remaining_time = max(0, self.frame_delay / 1000 - elapsed_time)  # Convert frame_delay to seconds
                if remaining_time > 0:
                    time.sleep(remaining_time)

    def toggle_pause(self):
        print("Toggling pause")
        self.paused = not self.paused
        print(f"Paused: {self.paused}")
        # # FIXME: handle pause frames preferable something like update_frame()

    def seek_to_frame(self, frame_number):
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._data.update_curr_frame(frame_number)
        

    def close_video_player(self):
        self.cap.release()
        self._data.save_data()
        cv2.destroyAllWindows()
        # TODO : close audio stream


class ControlWindow(ctk.CTkToplevel):
    def __init__(self, app, file_name, video_player):
        super().__init__(app)
        self.video_player = video_player
        self.title(f"Video Controls: {file_name}")
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=10)
        
        self.play_pause_button = ctk.CTkButton(self.control_frame, text="▐▐", command=self.toggle_pause)
        self.play_pause_button.grid(row=0, column=0, padx=5)
        
        self.seek_var = DoubleVar()
        self.seeker = ctk.CTkSlider(self.control_frame, variable=self.seek_var, from_=0, to=int(self.video_player.cap.get(cv2.CAP_PROP_FRAME_COUNT)), command=self.seek)
        self.seeker.grid(row=0, column=1, padx=5, sticky="ew")

        self.columnconfigure(1, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.close_control_window)
        
        # Make the control window screen topmost
        self.attributes("-topmost", True)

        print("Setting up control window...")

        self.video_player.video_thread.start()

        self.mainloop()

    def add_to_queue(self, command):
        self.video_player.command_queue.put(command)

    def toggle_pause(self):
        # print("Toggling pause")
        self.play_pause_button.configure(text="▶" if self.video_player.paused else "||")
        self.add_to_queue('pause')

    def seek(self, value):
        print(f"Seeking to frame {value}")
        frame_number = int(value)
        self.add_to_queue(('seek', frame_number))
        self.seek_var.set(frame_number)

    def close_control_window(self):
        self.video_player.close_video_player()
        self.video_player.done_event.set()
        self.destroy()