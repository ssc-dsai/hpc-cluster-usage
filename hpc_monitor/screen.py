import numpy as np
import os

class Display:
    """Display the cluster usage to the screen."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) # just put em all as attributes why not?
        self._width, self._height = os.get_terminal_size()
        self._screen = np.empty((self._height, self._width), dtype='U36')

    def __repr__(self):
        return repr(self._screen)

    def __str__(self):
        return repr(self._screen)
