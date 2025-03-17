from collections import OrderedDict
import getpass
import subprocess
from .sqstat import sinfof, squeuef, sinfof_local, squeuef_local

class ClusterStat:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs) # just put em all as attributes why not?
        self.users = OrderedDict()
        self.usercodes = OrderedDict() # colors for users

    @property
    def squeue(self):
        if not hasattr(self, "_squeue"):
            self._squeue = squeuef(self.clusters)
        return self._squeue

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
        for cluster in self.squeue.keys():
            for job in self.squeue[cluster]['jobs']:
                user = job['user_name']
                self.users.update({user: 0})
                if job['job_state'][0] == "RUNNING":
                    self.process_running_job(job)
                else:
                    pass

    def process_gpu_usage(self):
        pass
    
    def process_cpu_usage(self):
        pass

    def process_running_job(self, job):
        user = job['user_name']
        resources = job['job_resources']['nodes']
        node_names = [r_node['name'] for r_node in resources['allocation']]

        for r_node in resources['allocation']:
            self.process_cpu_usage()

        for gpuid, gpus_string in enumerate(job['gres_detail']):
            self.process_gpu_usage()

    def __call__(self):
        """Calling the cluster stat instance because I couldn't think of a good name for the function other 
        than `cluster_stat` which is redundant."""

        self.process_jobs()
        self.initialize_usercodes()
        print(self.usercodes)



