from collections import OrderedDict
import getpass
import subprocess
from .sqstat import sinfof, squeuef, sinfof_local, squeuef_local

class ClusterStat:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.users = OrderedDict()
        self.usercodes = OrderedDict() # colors for users
        self.clusters = args.clusters

    @property
    def squeue(self):
        if hasattr(self, _squeue):
            return self._squeue
        else:
            self._squeue = squeuef(self.clusters)

    def initialize_usercodes(self):
        """Initialize user codes for coloring output."""
        for user_id, user in enumerate(self.users, start=1):
            # Cycle through 15 colours (avoid black)
            user_id = user_id%15
            self.usercodes[user] = f"\033[38;5{user_id}m"
        # keep the active user the same color always
        self.usercodes[getpass.getuser()] = "\033[104;39;1m"

    def process_jobs(self):
        """Process jobs from squeue output."""
        pass

    def __call__(self):
        """Calling the cluster stat instance because I couldn't think of a good name for the function other 
        than `cluster_stat` which is redundant."""

        self.initialize_usercodes()
        self.process_jobs()



