# General Imports
import os, time, logging
from tkinter import messagebox
from customtkinter import filedialog

# Set up logging
logger = logging.getLogger('app')

class Config:
    def __init__(self, cwd=None, in_path=None, out_path=None, files=[], extension=".mp4"):
        self._cwd = cwd
        self._in_path = in_path
        self._out_path = out_path
        self._files = files
        self._last_update = time.time()
        self.extension = extension
        logger.info(f"Config initialized with cwd={self._cwd}, in_path={self._in_path}, out_path={self._out_path}")

    # METHODS TO GET DATA
    
    def __str__(self):
        return f"Config: cwd={self.cwd}, in_path={self.in_path}, out_path={self.out_path}, files={self.files}, last_update={self.last_update}"
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def cwd(self):
        return self._cwd

    @property
    def in_path(self):
        return self._in_path
    
    @property
    def files(self):
        return self._files
    
    @property
    def out_path(self):
        return self._out_path
    
    @property
    def last_update(self):
        return self._last_update

    @property
    def fetch_top_file(self):
        if len(self.files) == 0:
            return None
        return self.files[0]

    # METHODS TO UPDATE

    @cwd.setter
    def cwd(self, value):
        logger.debug(f"Setting Current Working Directory: {value}")
        self._cwd = value

    @in_path.setter
    def in_path(self, value):
        logger.debug(f"Setting input folder path to: {value}")
        self._in_path = value

    @out_path.setter
    def out_path(self, value):
        logger.debug(f"Setting output folder path to: {value}")
        self._out_path = value

    @files.setter
    def files(self, value):
        logger.debug(f"Setting Files: {value}")
        self._files = value

    @last_update.setter
    def last_update(self, value):
        logger.debug(f"Setting last_update: {time.time()}")
        self._last_update = time.time()

    def update(self, in_path=None, out_path=None, files=[]):
        self._cwd = os.getcwd()
        self._in_path = in_path
        self._out_path = out_path
        self._files = files
        self._last_update = time.time()
        logger.info(f"Config updated with in_path={self.in_path}, out_path={self.out_path}")

    def change_directory(self):
        logger.info("Starting directory change")

        try:
            new_dir = ""
            while len(new_dir) == 0:  new_dir = filedialog.askdirectory(title="Select New Directory")
            if new_dir:
                os.chdir(new_dir)
                logger.info(f"Current working directory changed to: {new_dir}")
                self.cwd = new_dir      # Update the config
                in_path, out_path, files = self.file_setup()
                return in_path, out_path, files, new_dir
            else:
                logger.warning("No directory selected for change")
        except Exception as e:
            logger.error(f"An error occurred during directory change: {e}", exc_info=True)

    def file_setup(self):
        logger.info("Starting file setup")

        try:
            if not self.cwd: self.cwd = os.getcwd()
            logger.debug(f"Current working directory: {self.cwd}")
            
            # Select Input Video Directory
            if not os.path.exists(f"{self.cwd}\data"): 
                self.in_path = ""
                while len(self.in_path) == 0: self.in_path = filedialog.askdirectory(title="Select Input Directory", initialdir=self.cwd)
                logger.info(f"Input directory selected: {self.in_path}")
            else: 
                self.in_path = f"{self.cwd}\data"
                logger.debug(f"Default input directory used: {self.in_path}")

            # Select Output Directory
            if not os.path.exists(f"{self.cwd}\out"): 
                self.out_path = ""
                while len(self.out_path) == 0: self.out_path = filedialog.askdirectory(title="Select Output Directory", initialdir=self.cwd)
                logger.info(f"Output directory selected: {self.out_path}")
            else: 
                self.out_path = f"{self.cwd}\out"
                logger.debug(f"Default output directory used: {self.out_path}")

            # get list of names all mp4 files in self.file_path
            in_files = [f for f in os.listdir(self.in_path) if f.endswith(".mp4")]
            logger.info(f"Found {len(in_files)} MP4 files in the input directory")

            if len(in_files) == 0: 
                logger.warning("No MP4 files found in the input directory")
                messagebox.showerror("Error", "No MP4 files found in the input directory.")
            
            files = config.refetch_files()

            return self.in_path, self.out_path, self.files
        except Exception as e:
            logger.error(f"An error occurred during file setup: {e}", exc_info=True)

    # METHODS TO EQUATE
    
    def __eq__(self, other):
        equality = self.cwd == other.cwd and self.in_path == other.in_path and self.out_path == other.out_path and self.files == other.files and self.last_update == other.last_update
        return equality

    # HELPER FUBNCTIONS
    
    def remove_extension(self, file_name, extensions=[".mp4"]):
        logger.debug(f"Removing extensions from file '{file_name}'")
        for ext in extensions:
            if file_name.endswith(ext):
                return file_name[:-len(ext)]
        logger.debug(f"No extension removed from file '{file_name}'")
        return file_name
    
    def refetch_files(self, inp=None, out=None, extensions=[".mp4"]):
        try:
            if inp is not None: in_files = [self.remove_extension(f, extensions) for f in os.listdir(inp) if f.endswith(tuple(extensions))]
            else: in_files = [self.remove_extension(f, extensions) for f in os.listdir(self.in_path) if f.endswith(tuple(extensions))]

            if out is not None: out_files = [self.remove_extension(f, extensions) for f in os.listdir(out) if f.endswith(tuple(extensions))]
            else: out_files = [self.remove_extension(f, extensions) for f in os.listdir(self.out_path) if f.endswith(tuple(extensions))]

            files = [f"{f}{self.extension}" for f in in_files if f"{f}_annotated" not in out_files]

            self.update(self.in_path, self.out_path, files)
            logger.info(f"Refetched files: {files}")
            return files
        except Exception as e:
            logger.error(f"Error refetching files: {e}")
            raise e

    # METHODS TO SAVE

config = Config()
