import re
import sys
from collections import OrderedDict
import numpy as np

import json
import argparse
import getpass
import subprocess
from .sqstat import sinfof, squeuef, sinfof_local, squeuef_local, job_smi
from .screen import Display

class ClusterStat:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs) # just put em all as attributes why not?
        self.users = OrderedDict()
        self.resource_gpu = OrderedDict()
        self.resource_gpu_desc = OrderedDict()
        self.resource_list = OrderedDict()
        self.resource_desc = OrderedDict()
        self.node_data = OrderedDict()

    @property
    def squeue(self):
        if not hasattr(self, "_squeue"):
            self._squeue = squeuef(self.clusters)
            #self._squeue = squeuef_local(self.clusters)
        return self._squeue
    
    @property
    def sinfo(self):
        if not hasattr(self, "_sinfo"):
            self._sinfo = sinfof(self.clusters)
            #self._sinfo = sinfof_local(self.clusters)
        return self._sinfo

    def parse_alloc_string(self, string):
        cpus = re.search(r'cpu=(\d+)', string, re.IGNORECASE)
        mem = re.search(r'(?<=mem=)[^,]*', string, re.IGNORECASE).group(0)
        nodes = re.search(r'node=(\d+)', string, re.IGNORECASE)
        gpus = re.search(r'gres/gpu=(\d+)', string, re.IGNORECASE)
        cpus = int(cpus.group(1)) if cpus else 0
        #mem = int(mem.group(1)) if mem else 0
        nodes = int(nodes.group(1)) if nodes else 0
        gpus = int(gpus.group(1)) if gpus else 0
        n_mem = 0
        # convert to megabytes
        if mem.endswith("T"):
            n_mem = int(float(mem[:-1])) * 1024 * 1024 
        elif mem.endswith("G"):
            n_mem = int(float(mem[:-1])) * 1024
        elif mem.endswith("M"):
            n_mem = int(float(mem[:-1]))
        return {"cpus": cpus, "gpus": gpus, "mem": n_mem, "nodes": nodes}


    def process_jobs(self):
        """Process jobs from squeue output.
        
        Possible job states are:
                    (BOOT_FAIL, 
                     CANCELLED, 
                     COMPLETED, 
                     DEADLINE, 
                     FAILED,
                     NODE_FAIL, 
                     OUT_OF_MEMORY, 
                     PENDING, 
                     PREEMPTED, 
                     RUNNING
                     SUSPENDED, 
                     TIMEOUT,
        )
        """
        for cluster in self.squeue.keys():
            for job in self.squeue[cluster]['jobs']:
                self.users.setdefault(cluster, OrderedDict())
                user = job['user_name']
                self.users[cluster].setdefault(user, OrderedDict())
                group_name = job['group_name']
                self.users[cluster][user]['group'] = group_name 
                # below branch stuff is very HPCO specific.
                branch = group_name.split("_")[0].upper()
                self.users[cluster][user]['branch'] = branch 
                alloc = self.parse_alloc_string(job['tres_req_str'])
                if job['job_state'][0] == "RUNNING":
                    self.users[cluster][user].setdefault("RUNNING", [])
                    self.users[cluster][user]["RUNNING"].append(alloc)
                    self.process_running_job(job)
                elif job['job_state'][0] == "PENDING":
                    self.users[cluster][user].setdefault("PENDING", [])
                    self.users[cluster][user]["PENDING"].append(alloc)
                else:
                    print(f"{job['job_state'][0]} not considered yet!")

    def process_gpu_usage(self, node_name, gpus_string, n_nodes, cluster, user):
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

        if (len(gpu_range) == 1) and (n_nodes <= 1):
            self.resource_gpu[cluster][node_name][gpu_name][gpu_range] += 1
            self.resource_gpu_desc[cluster][node_name][gpu_name][gpu_range] = f"S{user}" #f"{self.usercodes[user]}+\033[0m"
        else: 
            self.resource_gpu[cluster][node_name][gpu_name][gpu_range] += 1
            self.resource_gpu_desc[cluster][node_name][gpu_name][gpu_range] = f"P{user}" #f"{self.usercodes[user]}=\033[0m"
    
    def process_cpu_usage(self, r_node, n_nodes, cluster, user):
        r_node_name = r_node['name']
        cores_array = self.resource_list[cluster][r_node_name]
        desc_array = self.resource_desc[cluster][r_node_name]
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
        if (cpu_usage <= 1) and (n_nodes <= 1):
            # need user data as well!
            desc_array[usage_idx] = f"S{user}" #f"{self.usercodes[user]}+\033[0m"
        else:
            desc_array[usage_idx] = f"P{user}" #f"{self.usercodes[user]}=\033[0m"

    def process_info(self):
        for cluster, sinfo_list in self.sinfo.items():
            for info in sinfo_list['sinfo']:
                node_name = info['nodes']['nodes'][0]
                # state info['node']['state'] = ["DOWN", "DRAIN", "NOT_RESPONDING"]
                cores_per_socket = info['cores']['maximum']
                n_sockets = info['sockets']['maximum']
                n_cpus = int(cores_per_socket) * int(n_sockets)
                gpus_string = info['gres']['total']
                gpu_name = None
                gpu_count = 0
                node_states = info['node']['state']
                down_node = False
                if ("DOWN" in node_states) or ("DRAIN" in node_states) or ("NOT_RESPONDING" in node_states):
                    down_node = True

                if gpus_string:
                    gpus_list = re.split(r'[:()\s]+', gpus_string)
                    gpu_name = gpus_list[1]
                    gpu_count = int(gpus_list[2])
                    self.resource_gpu.setdefault(cluster, OrderedDict())
                    self.resource_gpu_desc.setdefault(cluster, OrderedDict())
                    self.resource_gpu[cluster].setdefault(node_name, OrderedDict())
                    self.resource_gpu_desc[cluster].setdefault(node_name, OrderedDict())
                    self.resource_gpu[cluster][node_name].setdefault(gpu_name, np.zeros(gpu_count, dtype=object))
                    self.resource_gpu_desc[cluster][node_name].setdefault(gpu_name, np.zeros(gpu_count, dtype=object))
                    if down_node:
                        self.resource_gpu[cluster][node_name][gpu_name].fill(-1)
                        self.resource_gpu_desc[cluster][node_name][gpu_name].fill(-1)

                self.resource_list.setdefault(cluster, OrderedDict())
                self.resource_desc.setdefault(cluster, OrderedDict())
                self.resource_list[cluster].setdefault(node_name, np.zeros(n_cpus, dtype=object))
                self.resource_desc[cluster].setdefault(node_name, np.zeros(n_cpus, dtype=object))
                if down_node:
                    self.resource_desc[cluster][node_name].fill(-1)
                    self.resource_desc[cluster][node_name].fill(-1)
            
                self.node_data.setdefault(cluster, OrderedDict())
                self.node_data[cluster][node_name] = {
                        'memory': info['memory']['maximum'],
                        'memory_used': info['memory']['allocated'],
                        'cpu_count': n_cpus,
                        'cpu_used': info['cpus']['allocated'],
                        'cpu_idle': info['cpus']['idle'],
                        'core_count': info['cores']['maximum'],
                        'gpu_name': gpu_name,
                        'gpu_count': gpu_count,
                }

    def process_running_job(self, job):
        user = job['user_name']
        cluster = job['cluster']
        resources = job['job_resources']['nodes']
        node_names = [r_node['name'] for r_node in resources['allocation']]
        for r_node in resources['allocation']:
            self.process_cpu_usage(r_node, len(resources['allocation']), cluster, user)

        for gpuid, gpus_string in enumerate(job['gres_detail']):
            self.process_gpu_usage(node_names[gpuid], gpus_string, len(job['gres_detail']), cluster, user)

    def __call__(self):
        """Calling the cluster stat instance because I couldn't think of a good name for the function other 
        than `cluster_stat` which is redundant."""

        self.process_info() # must be called first
        self.process_jobs() # second..


def parse_job_args():
    parser = argparse.ArgumentParser(description="Graphical representation of GPU usage for a job.")
    parser.add_argument('jobid', type=int, help='Specify the SLURM jobid for the GPUs you wish to see.')
    parser.add_argument('--cluster', '-M', default='all', help='Specify the cluster name the job is running on.')
    parser.add_argument('--local', action='store_true', help='Tell the program to search the directory for GPU usage files.')
    return parser.parse_args()


def parse_cs_args():
    parser = argparse.ArgumentParser(description="Graphical representation of HPC usage.")
    parser.add_argument('--clusters', '-M', default='all', help='Specify the cluster to display on screen.')
    parser.add_argument('--gpus-only', '-g', action='store_true', help='Print out only the GPU nodes.')
    return parser.parse_args()

def job_main():
    args = parse_job_args()
    args_dict = vars(args)
    screen = Display(**args_dict)
    screen.print_gpu_usage()

def main():
    args = parse_cs_args()
    args_dict = vars(args)
    cs = ClusterStat(**args_dict)
    cs()
    screen = Display(**args_dict)
    screen.cluster_stat = cs
    screen.initialize_usercodes()
    print(screen)

if __name__ == '__main__':
    main()

