import cv2, os
from PIL import Image
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import messagebox
from data import Data

class VideoPlayer:
    def __init__(self, file_path, file_name, out_path):
        # TODO : add audio callback function
        # Video Player Objects
        self.cap = None

        # Variables
        self.file_path, self.file_name = file_path, file_name
        self.cwd, self.out_path = os.getcwd(), out_path
        self._data = None
        self.last_frame, self.last_point = None, None
        self.wait_between_frames = 0

        # FLags
        self.paused, self.drawing = False, False

        # widgets
        self.video_window, self.video_label = None, None
        self.control_frame = None
        self.play_pause_button, self.seeker = None, None

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
        self.wait_between_frames = int(1000 // self._data.FPS - 1)

        print(f"FPS recorded: {self._data.FPS}")
        print(f"Frame Count: {self._data.frame_count}")
        print(f"Wait Time: {self.wait_between_frames}")

        # TODO: setup audio input

        self.setup_video_window()
        self.update_frame()

    def setup_video_window(self):
        self.video_window = ctk.CTkToplevel()
        self.video_window.title("Video Player")

        # Handle window close event
        self.video_window.protocol("WM_DELETE_WINDOW", lambda: self.close_window(self.video_window))
        
        # Create a label to display the video
        self.video_label = ctk.CTkLabel(self.video_window, text="")
        self.video_label.pack()

        # Bind mouse events
        self.video_label.bind("<Button>", self.on_mouse_click)
        self.video_label.bind("<B1-Motion>", self.on_mouse_move)

        # Create a frame to hold the controls
        self.control_frame = ctk.CTkFrame(self.video_window)
        self.control_frame.pack(pady=10)

        # Handle Play/Pause button and Pause frames
        self.play_pause_button = ctk.CTkButton(self.control_frame, text="▐▐", command=self.toggle_pause)
        self.play_pause_button.grid(row=0, column=0, padx=5)

        # Handle Seeker bar
        self.seeker = ctk.CTkSlider(self.control_frame, from_=0, to=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), command=self.seek)
        self.seeker.grid(row=0, column=1, padx=5, sticky="ew")

        # Configure the grid
        self.video_window.columnconfigure(1, weight=1)
    
    def close_window(self, video_window):
        self.cap.release()
        self._data.save_data()
        # self.audio_stream.stop()
        video_window.destroy()

    def on_mouse_click(self, event):
        if event.num == 1 and self.paused:  # Ensure this happens only if the video is paused
            self.drawing = True
            self.last_point = (event.x, event.y)
        elif not self.paused:
            self.drawing = False

    def on_mouse_move(self, event):
        if self.drawing and self.paused:
            x, y = event.x, event.y
            # TODO : implement draw_annotation later on in seperate dictionary
            self._data.add_annotation(((x, y), self.last_point))
            self.last_point = (x, y)

    def toggle_pause(self):
        self.paused = not self.paused

        # Update the label of the button based on the current state after toggling
        self.play_pause_button.configure(text="▶" if self.paused else "▐▐")
        # FIXME: handle pause frames preferable something like update_frame()

    def seek(self, value):
        # TODO : Make this faster and in real time
        print(f"Seeking to frame {value}")
        self._data.update_curr_frame(int(value))
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
        # self.update_frame()

    def update_frame(self):
        if not self.paused:
            ret, frame = self.cap.read()
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Current Frame Number: {current_frame}")
            if ret:     # if frame is read successfully
                self._data.add_curr_frame(current_frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # convert the frame to RGB
                img = Image.fromarray(frame)                            # convert the frame to an image
                ctk_img = CTkImage(light_image=img, size=(frame.shape[1], frame.shape[0]))  # create a custom image object
                
                self.video_label.configure(image=ctk_img)                # update the label
                self.video_label.image = ctk_img                         # update the image

                self.seeker.set(current_frame)
                self.video_label.after(self.wait_between_frames, self.update_frame)
                
            else:       # if frame is not read successfully
                self.cap.release()
                self._data.save_data()
                # self.audio_stream.stop()
                self.video_window.destroy()
