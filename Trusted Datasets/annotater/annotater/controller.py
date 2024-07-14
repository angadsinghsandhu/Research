
# General imports
import cv2
import customtkinter as ctk
from tkinter import DoubleVar

# TODO : Make seeker faster and in real time

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
        self.seeker = ctk.CTkSlider(self.control_frame, variable=self.seek_var, from_=0, to=int(self.video_player.cap.get(cv2.CAP_PROP_FRAME_COUNT)), command=self.seek)
        self.seeker.grid(row=0, column=1, padx=5, sticky="ew")

        self.restart_button = ctk.CTkButton(self.control_frame, text="↺", command=self.restart)
        self.restart_button.grid(row=0, column=2, padx=5)

        self.stop_button = ctk.CTkButton(self.control_frame, text="■", command=self.save_and_close)
        self.stop_button.grid(row=0, column=3, padx=5)

        self.columnconfigure(1, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.close)
        
        # Make the control window screen topmost
        self.attributes("-topmost", True)

    def add_to_queue(self, command):
        self.video_player.command_queue.put(command)

    def toggle_pause(self):
        self.play_pause_button.configure(text="▶" if self.video_player.paused else "||")
        self.add_to_queue('pause')

    def seek(self, value):
        frame_number = int(value)
        self.add_to_queue(('seek', frame_number))
        self.seek_var.set(frame_number)

    def restart(self):
        self.add_to_queue('restart')
        self.play_pause_button.configure(text="▐▐")
        self.seek_var.set(0)

    def save_and_close(self):
        self.video_player.close(save=True)
        self.video_player.done_event.set()

    def close(self):
        self.video_player.close()
        self.video_player.done_event.set()