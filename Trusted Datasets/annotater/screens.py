# General Imports
import os, customtkinter as ctk
from PIL import Image

class Splash(ctk.CTkToplevel):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.create_splash()
        self.bind("<Configure>", self.update_window_position)
        self.update_countdown(3)

    def create_splash(self):
        self.title("Loading...")

        # Center the splash screen
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.window_width = int(self.screen_width // 3)
        self.window_height = int(self.screen_height // 2)
        self.position_top = (self.screen_height // 2) - (self.window_height // 2)
        self.position_right = (self.screen_width // 2) - (self.window_width // 2)
        self.geometry(f"{self.window_width}x{self.window_height}+{self.position_right}+{self.position_top}")

        # Make the splash screen topmost
        self.attributes("-topmost", True)

        # Load and resize the image
        self.image_path = "./imgs/jhu.png"
        self.max_image_width = self.window_width - 40  # Max width for the image with padding
        self.max_image_height = self.window_height // 2  # Max height for the image

        if os.path.exists(self.image_path):
            img = Image.open(self.image_path)
            img_width, img_height = img.size

            # # Debug print statement
            # print(f"Original Image Size: {img_width}x{img_height}")

            # Scale down the image if it is too big
            if img_width > self.max_image_width or img_height > self.max_image_height:
                scaling_factor = min(self.max_image_width / img_width, self.max_image_height / img_height)
                new_width = int(img_width * scaling_factor)
                new_height = int(img_height * scaling_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            #     # Debug print statement
            #     print(f"Resized Image Size: {new_width}x{new_height}")

            # print(f"Image size: {img.size}")
            # img_ctk = ImageTk.PhotoImage(img)  # Convert to PhotoImage
            img_ctk = ctk.CTkImage(img, size=(img.width, img.height))  # Convert to CTkImage
            # Create and place the image label
            self.image_label = ctk.CTkLabel(self, text="", image=img_ctk)
            self.image_label.pack(pady=20)
        else:
            print(f"Image not found: {self.image_path}")

        # Create and place the text label
        self.label = ctk.CTkLabel(self, text="Welcome to the Annotater Application", font=("Arial", 16))
        self.label.pack(pady=10)

        self.countdown_label = ctk.CTkLabel(self, text="Closing in 3 seconds", font=("Courier", 12))
        self.countdown_label.pack(pady=10)

        self.window_position_label = ctk.CTkLabel(self, text=f"Window Position: {self.winfo_x()}, {self.winfo_y()}")
        self.window_position_label.pack(pady=2)

    def update_countdown(self, count):
        if count > 0:
            self.countdown_label.configure(text=f"Closing in {count} seconds...")
            self.after(1000, self.update_countdown, count-1)
        else:
            self.destroy_splash()

    def update_window_position(self, event):
        self.window_position_label.configure(text=f"Window Position: {self.winfo_x()}, {self.winfo_y()}")

    def destroy_splash(self):
        self.destroy()
        self.root.deiconify()    # Show the main window


class SaveProgress(ctk.CTkToplevel):
    def __init__(self, root, name):
        super().__init__(root)
        self.root = root
        self.name = name
        self.create_save_progress()
        self.protocol("WM_DELETE_WINDOW", self.destroy_save_progress)

    def create_save_progress(self):
        self.title(f"{self.name}: Saving Progress...")
        self.geometry("400x250")
        self.attributes("-topmost", True)

        # video data progress bar
        self.video_progress = ctk.CTkProgressBar(self, mode='determinate')
        self.video_progress.pack(pady=10)
        self.video_progress_label = ctk.CTkLabel(self, text="Video Data Save Progress...")
        self.video_progress_label.pack(pady=5)

        # audio data progress bar
        self.audio_progress = ctk.CTkProgressBar(self, mode='determinate')
        self.audio_progress.pack(pady=10)
        self.audio_progress_label = ctk.CTkLabel(self, text="Audio Data Save Progress...")
        self.audio_progress_label.pack(pady=5)

        # audio-video data progress bar
        self.av_progress = ctk.CTkProgressBar(self, mode='determinate')
        self.av_progress.pack(pady=10)
        self.av_progress_label = ctk.CTkLabel(self, text="Audio-Video Data Save Progress...")
        self.av_progress_label.pack(pady=5)

        # annotations json data progress bar
        self.json_progress = ctk.CTkProgressBar(self, mode='determinate')
        self.json_progress.pack(pady=10)
        self.json_progress_label = ctk.CTkLabel(self, text="JSON Data Save Progress...")
        self.json_progress_label.pack(pady=5)

        # reset progress bars
        self.reset()

    def update_title_on_save(self):
        self.title(f"{self.name}: Progress Saved!!!")

    def update_video_progress(self, value):
        self.video_progress.set(value)

    def update_audio_progress(self, value):
        self.audio_progress.set(value)

    def update_av_progress(self, value):
        self.av_progress.set(value)

    def update_json_progress(self, value):
        self.json_progress.set(value)

    def reset(self):
        self.title(f"{self.name}: Saving Progress...")
        self.update_video_progress(0.0)
        self.update_audio_progress(0.0)
        self.update_av_progress(0.0)
        self.update_json_progress(0.0)

    def destroy_save_progress(self):
        self.destroy()
        self.root.deiconify()    # Show the main window

