import numpy as np
import os
import getpass
from collections import OrderedDict

class Display:
    """Display the cluster usage to the screen."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) # just put em all as attributes why not?
        self._width, self._height = os.get_terminal_size()
        self._screen = np.empty((self._height, self._width), dtype='U36')
        self._usercodes = OrderedDict()
        self._cluster_stat = None

    @property
    def cluster_stat(self):
        return self._cluster_stat

    @cluster_stat.setter
    def cluster_stat(self, val):
        self._cluster_stat = val

    def initialize_usercodes(self):
        if self._cluster_stat is None:
            print(f"Warning: initialize_usercodes was called but cluster_stat was not defined!")
            return

        for user_id, user in enumerate(self.cluster_stat.users, start=1):
            user_id = user_id%15
            self._usercodes[user] = f"\033[38;5{user_id}m"
        # keep the active user the same color
        self._usercodes[getpass.getuser()] = "\033[104;39;1m"


    def __repr__(self):
        return repr(self._screen)

    def __str__(self):
        return repr(self._screen)
