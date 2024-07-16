# General Imports
import logging, logging.config, yaml, ctypes
import customtkinter as ctk
from screens import Splash

# Custom Imports
from annotater.anno import create_annotater

# TODO : increase length of slider in control window
# TODO : add final touches to UI (annotater and save Data window)
# TODO : add comments to everything
# TODO : stress testing
# TODO : create documentation
# TODO : test Surface audio signal
#  : test annotations recorded 240p and recoreded in 1080p, scaled down to 240p

# Set DPI Awareness for Windows 10
ctypes.windll.shcore.SetProcessDpiAwareness(2)

# Set up logging
def setup_logging(config_path="logging_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config=config)
    logger = logging.getLogger('main')
    return logger

logger = setup_logging()

def close(app):
    logger.info("### Closing the Annotation Tool ###")
    app.destroy()

# run main loop
if __name__ == "__main__":
    logger.info("### Starting the Annotation Tool ###")

    try:
        # Set the theme (optional)
        ctk.set_appearance_mode("Dark")  # Can be "Dark" or "Light"
        logger.debug("Appearance mode set to Dark")

        # Create the main application window
        app = ctk.CTk()
        app.title("Annotater") # Set the title of the window
        app.protocol("WM_DELETE_WINDOW", lambda: close(app))
        logger.debug("Main application window created and configured")

        # add calback for mouse position
        app.bind("<Motion>", lambda e: logger.debug(f"Mouse position: {e.x}, {e.y}"))

        # Hide the main application window initially
        app.withdraw()
        logger.debug("Main application window hidden")

        # Show splash screen
        splash = Splash(app)
        logger.info("Splash screen displayed")

        create_annotater(app)
        logger.info("Annotater created and started")

    except Exception as e: logger.exception("An error occurred in the main loop: %s", e)
    finally: logger.info("Application terminated")
