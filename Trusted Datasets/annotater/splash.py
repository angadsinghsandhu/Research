# General Imports
import os, customtkinter as ctk
from PIL import Image

def show_splash(root):
    # FIXME : understand why image is not showing on splash screen
    splash = ctk.CTkToplevel(root)
    splash.title("Loading...")

    # Center the splash screen
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    window_width = int(screen_width//3)
    window_height = int(screen_height//2)
    position_top = (screen_height // 2) - (window_height // 2)
    position_right = (screen_width // 2) - (window_width // 2)
    splash.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    # Make the splash screen topmost
    splash.attributes("-topmost", True)

    # Load and resize the image
    image_path = "./imgs/jhu.png"
    max_image_width = window_width - 40  # Max width for the image with padding
    max_image_height = window_height // 2  # Max height for the image

    if os.path.exists(image_path):
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Debug print statement
        print(f"Original Image Size: {img_width}x{img_height}")

        # Scale down the image if it is too big
        if img_width > max_image_width or img_height > max_image_height:
            scaling_factor = min(max_image_width / img_width, max_image_height / img_height)
            new_width = int(img_width * scaling_factor)
            new_height = int(img_height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Debug print statement
            print(f"Resized Image Size: {new_width}x{new_height}")

        print(f"Image size: {img.size}")
        # img_ctk = ImageTk.PhotoImage(img)  # Convert to PhotoImage
        img_ctk = ctk.CTkImage(img, size=(img.width, img.height))  # Convert to CTkImage
        # Create and place the image label
        image_label = ctk.CTkLabel(splash, text="", image=img_ctk)
        image_label.pack(pady=20)
    else:
        print(f"Image not found: {image_path}")

    # Create and place the text label
    label = ctk.CTkLabel(splash, text="Welcome to the Annotater Application", font=("Arial", 16))
    label.pack(pady=10)

    countdown_label = ctk.CTkLabel(splash, text="Closing in 3 seconds", font=("Courier", 12))
    countdown_label.pack(pady=10)

    def update_countdown(count):
        if count > 0:
            countdown_label.configure(text=f"Closing in {count} seconds")
            splash.after(1000, update_countdown, count-1)
        else:
            splash.destroy()
            root.deiconify()    # Show the main window

    window_position_label = ctk.CTkLabel(splash, text=f"Window Position: {splash.winfo_x()}, {splash.winfo_y()}")
    window_position_label.pack(pady=2)

    # update window position label on window move
    def update_window_position(event):
        window_position_label.configure(text=f"Window Position: {splash.winfo_x()}, {splash.winfo_y()}")

    splash.bind("<Configure>", update_window_position)

    splash.update()
    update_countdown(3)