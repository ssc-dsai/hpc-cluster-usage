import re
from collections import OrderedDict
import getpass
import subprocess
from .sqstat import sinfof, squeuef, sinfof_local, squeuef_local

class ClusterStat:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs) # just put em all as attributes why not?
        self.users = OrderedDict()
        self.usercodes = OrderedDict() # colors for users
        self.resource_gpu = OrderedDict()
        self.resource_gpu_desc = OrderedDict()
        self.resource_list = OrderedDict()
        self.resource_desc = OrderedDict()

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

    def process_gpu_usage(self, node_name, gpus_string, user):
        name_qty = re.sub(r'\([^)]*\)', '', gpus_string)
        gpu_name = name_qty.split(":")[1]
        n_gpus = int(name_qty.split(":")[-1])
        gpu_idx = re.search(r'\((.*?)\)', gpus_string).group(1).split(":")[-1]
        gpu_range = []

        for i in gpu_idx.split(','):
            rangesplit = i.split("-")
            if len(rangesplit)==1:
                gpu_range.append(int(rangesplit[0]))
            else:
                range_i = [int(j) for j in rangesplit]
                range_i[-1] += 1
                gpu_range += list(range(*range_i))

        if (len(gpu_range) == 1) and (len(job['gres_detail']) == 1):
            self.resource_gpu[node_name][gpu_name][gpu_range] += 1
            self.resource_gpu_desc[node_name][gpu_name][gpu_range] = f"{self.usercodes[user]}+\033[0m"
        else: 
            self.resource_gpu[node_name][gpu_name][gpu_range] += 1
            self.resource_gpu_desc[node_name][gpu_name][gpu_range] = f"{self.usercodes[user]}=\033[0m"
    
    def process_cpu_usage(self, r_node, user):
        r_node_name = r_node['name']
        cores_array = self.resource_list[r_node_name]
        desc_array = self.resource_desc[r_node_name]
        offset = 0
        cpu_usage = 0
        usage_idx = []

        for socket in r_node['sockets']:
            for core in socket['cores']:
                if core['status'][0] == "ALLOCATED":
                    cores_array[offset] += 1
                    cpu_usage += 1
                    usage_idx.append(offset)
                offset += 1
        if cpu_usage > 1:
            desc_array[usage_idx] = f"{self.usercodes[user]}=\033[0m"
        else:
            desc_array[usage_idx] = f"{self.usercodes[user]}+\033[0m"


    def process_running_job(self, job):
        user = job['user_name']
        resources = job['job_resources']['nodes']
        node_names = [r_node['name'] for r_node in resources['allocation']]

        for r_node in resources['allocation']:
            self.process_cpu_usage(r_node, user)

        for gpuid, gpus_string in enumerate(job['gres_detail']):
            self.process_gpu_usage(node_names[gpuid], gpus_string, user)

    def __call__(self):
        """Calling the cluster stat instance because I couldn't think of a good name for the function other 
        than `cluster_stat` which is redundant."""

        self.process_jobs()
        self.initialize_usercodes()


