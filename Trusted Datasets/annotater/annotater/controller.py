
# General imports
import cv2, logging
import customtkinter as ctk
from tkinter import DoubleVar

# Custom imports
from annotater.setup import align_window

# Set up logging
logger = logging.getLogger('app')

class ControlWindow(ctk.CTkToplevel):
    def __init__(self, app, file_name, video_player, *args, **kwargs):
        super().__init__(app, *args, **kwargs)
        self.video_player = video_player
        self.title(f"Video Controls: {file_name}")

        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=10)
        
        self.play_pause_button = ctk.CTkButton(self.control_frame, text="▐▐", command=self.toggle_pause)
        self.play_pause_button.grid(row=0, column=0, padx=5)
        
        self.seek_var = DoubleVar()
        self.seeker = ctk.CTkSlider(self.control_frame, variable=self.seek_var, from_=0, to=int(self.video_player.cap.get(cv2.CAP_PROP_FRAME_COUNT)), width=500, command=self.seek)
        self.seeker.grid(row=0, column=1, padx=5, sticky="ew")

        self.restart_button = ctk.CTkButton(self.control_frame, text="⟳", command=self.restart)
        self.restart_button.grid(row=0, column=2, padx=5)

        self.stop_button = ctk.CTkButton(self.control_frame, text="⏹", command=self.save_and_close)
        self.stop_button.grid(row=0, column=3, padx=5)

        self.columnconfigure(1, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.close)

        # center the control window
        self.window_width, self.window_height = self.winfo_width(), self.winfo_height()
        print(f"==> Control Window Width: {self.window_width}, Control Window Height: {self.window_height}")

        # align the control window
        (self.mid_x, self.mid_y), (self.window_width, self.window_height), (self.screen_width, self.screen_height) = align_window(self, 970, 50, horizontal="left", vertical="bottom")
        
        # Make the control window screen topmost
        self.attributes("-topmost", True)
        logger.info(f"Control window configuration completed for file: {file_name}")

    def add_to_queue(self, command):
        self.video_player.command_queue.put(command)
        logger.debug("Added command to queue: %s", command)

    def toggle_pause(self):
        self.play_pause_button.configure(text="▶" if self.video_player.paused else "▐▐")
        self.add_to_queue('pause')
        logger.info("Toggled pause state to %s", not self.video_player.paused)

    def seek(self, value):
        frame_number = int(value)
        self.add_to_queue(('seek', frame_number))
        self.seek_var.set(frame_number)
        logger.info("Seeked to frame number: %d", frame_number)

    def restart(self):
        self.add_to_queue('restart')
        self.play_pause_button.configure(text="▐▐")
        self.seek_var.set(0)
        logger.info("Restarted video")

    def save_and_close(self):
        self.attributes("-topmost", False)
        self.video_player.close(save=True)
        self.video_player.done_event.set()
        logger.info("Saved and closed the video player")

    def close(self):
        self.video_player.close()
        self.video_player.done_event.set()
        logger.info("Closed the video player")