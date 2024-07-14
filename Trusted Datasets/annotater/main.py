# imports
import customtkinter as ctk
from splash import Splash
from annotater.anno import create_annotater

# TODO: Add draw annotations
# TODO: test Surface audio signal
# TODO: test annotations recorded 240p and recoreded in 1080p, scaled down to 240p

# run main loop
if __name__ == "__main__":

    # Set the theme (optional)
    ctk.set_appearance_mode("Dark")  # Can be "Dark" or "Light"

    # Create the main application window
    app = ctk.CTk()
    app.title("Annotater") # Set the title of the window
    app.protocol("WM_DELETE_WINDOW", lambda: app.destroy())

    # Hide the main application window initially
    app.withdraw()

    # Show splash screen
    splash = Splash(app)

    create_annotater(app)
