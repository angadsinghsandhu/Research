# General Imports
import os, time

class Config:
    def __init__(self, cwd=None, in_path=None, out_path=None, files=[], extension=".mp4"):
        self._cwd = cwd
        self._in_path = in_path
        self._out_path = out_path
        self._files = files
        self._last_update = time.time()
        self.extension = extension

    def update(self, in_path=None, out_path=None, files=[]):
        self.cwd = os.getcwd()
        self.in_path = in_path
        self.out_path = out_path
        self.files = files
        self.last_update = time.time()

    def __str__(self):
        return f"Config: cwd={self.cwd}, in_path={self.in_path}, out_path={self.out_path}, files={self.files}, last_update={self.last_update}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.cwd == other.cwd and self.in_path == other.in_path and self.out_path == other.out_path and self.files == other.files and self.last_update == other.last_update

    @property
    def fetch_top_file(self):
        if len(self.files) == 0:
            return None
        return self.files[0]
    
    def remove_extension(self, file_name, extensions=[".mp4"]):
        for ext in extensions:
            if file_name.endswith(ext):
                return file_name[:-len(ext)]
        return file_name
    
    def refetch_files(self, inp=None, out=None, extensions=[".mp4"]):
        if inp is not None: self.in_path = inp
        if out is not None: self.out_path = out

        in_path, out_path = self.in_path, self.out_path
        in_files = [self.remove_extension(f, extensions) for f in os.listdir(in_path) if f.endswith(tuple(extensions))]
        out_files = [self.remove_extension(f, extensions) for f in os.listdir(out_path) if f.endswith(tuple(extensions))]

        # files = [f for f in files if f"{f}" not in os.listdir(out_path) if f.endswith(".mp4")]
        files = [f"{f}{self.extension}" for f in in_files if f"{f}_annotated" not in out_files]

        self.update(in_path, out_path, files)
        return files

    # Properties with getters and setters
    @property
    def cwd(self):
        return self._cwd

    @cwd.setter
    def cwd(self, value):
        self._cwd = value
    
    @property
    def in_path(self):
        return self._in_path

    @in_path.setter
    def in_path(self, value):
        self._in_path = value
    
    @property
    def out_path(self):
        return self._out_path

    @out_path.setter
    def out_path(self, value):
        self._out_path = value
    
    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, value):
        self._files = value
    
    @property
    def last_update(self):
        return self._last_update

    @last_update.setter
    def last_update(self, value):
        self._last_update = value

config = Config()