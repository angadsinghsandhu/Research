# states.py
"""
Defines the various states of the application.
"""

from enum import Enum

class AppState(Enum):
    MAIN = 1
    SETTINGS = 2
    HELP = 3
    ABOUT = 4
    WEBCAM = 5
