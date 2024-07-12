import cv2, os, threading, queue, time
from PIL import Image
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import messagebox, Tk, DoubleVar
from data import Data

# TODO : add audio callback function
# TODO: setup audio input
# TODO : implement draw_annotation later on in seperate dictionary
# TODO : Make seeker faster and in real time

class VideoPlayer:
    def __init__(self, app, file_path, file_name, out_path):
        
        # Video Player Objects
        self.app = app
        self.cap = None

        # Variables
        self.file_path, self.file_name = file_path, file_name
        self.cwd, self.out_path = os.getcwd(), out_path
        self._data = None
        self.last_frame, self.last_point = None, None

        # FLags
        self.paused, self.drawing = False, False

        # widgets
        self.control_window, self.control_frame = None, None
        self.play_pause_button, self.seeker = None, None
        self.seek_var = DoubleVar()

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

        # TODO : 
        # self.setup_control_window()

        self.main_loop()

    def setup_control_window(self):
        self.control_window = ctk.CTkToplevel(self.app)
        self.control_window.title("Video Controls")

        self.control_frame = ctk.CTkFrame(self.control_window)
        self.control_frame.pack(pady=10)

        self.play_pause_button = ctk.CTkButton(self.control_frame, text="▐▐", command=self.toggle_pause)
        self.play_pause_button.grid(row=0, column=0, padx=5)

        self.seeker = ctk.CTkSlider(self.control_frame, variable=self.seek_var, from_=0, to=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), command=self.seek)
        self.seeker.grid(row=0, column=1, padx=5, sticky="ew")

        self.control_window.columnconfigure(1, weight=1)
        self.control_window.protocol("WM_DELETE_WINDOW", self.close_control_window)
        # self.control_window.mainloop()

    def close_control_window(self):
        self.cap.release()
        self._data.save_data()
        # self.audio_stream.stop()
        self.control_window.destroy()
        cv2.destroyAllWindows()

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

    def toggle_pause(self):
        self.paused = not self.paused

        # Update the label of the button based on the current state after toggling
        self.play_pause_button.configure(text="▶" if self.paused else "||")
        # FIXME: handle pause frames preferable something like update_frame()

    def seek(self, value):
        print(f"Seeking to frame {value}")
        frame_number = int(value)
        self._data.update_curr_frame(int(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def mouse_callback(self, event, x, y, flags, param):
        pass
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     self.paused = not self.paused

    def close_video_player(self):
        self.cap.release()
        self._data.save_data()
        cv2.destroyAllWindows()

    def main_loop(self):
        cv2.namedWindow("Video Player")
        cv2.setMouseCallback("Video Player", self.mouse_callback)
        # cv2.createButton("||", self.toggle_pause)
        # cv2.createButton("||", self.toggle_pause, ["Pause"], 1, 0)
        # cv2.createButton("⟳", self.reset ,None, ["Reset"], 1, 0)
        # cv2.createTrackbar("seeker", "Video Player", 0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.seek)
        # cv2.createButton("◼", self.close_video_player, ["Stop"], 1, 1)

        while self.cap.isOpened():
            start_time = time.time()
            if not self.paused:
                ret, frame = self.cap.read()
                curr_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"Current Frame: {curr_frame}")
                
                if not ret: break
                    
                self.last_frame = frame.copy()
                self._data.add_curr_frame(curr_frame)
                cv2.imshow("Video Player", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF  # Ensure responsiveness during pause
                if key == ord('q'): break

            elapsed_time = time.time() - start_time
            remaining_time = max(0, self.frame_delay / 1000 - elapsed_time)  # Convert frame_delay to seconds
            if remaining_time > 0:
                time.sleep(remaining_time)

        self.cap.release()
        self._data.save_data()
        cv2.destroyAllWindows()
        print("Window closed")
