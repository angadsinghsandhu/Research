import os, time

class Config:
    def __init__(self, cwd=None, in_path=None, out_path=None, files=[]):
        self._cwd = cwd
        self._in_path = in_path
        self._out_path = out_path
        self._files = files
        self._last_update = time.time()

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
