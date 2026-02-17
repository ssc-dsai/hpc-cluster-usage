import re
import sys
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import time

import json
import argparse
import getpass
import subprocess
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .sqstat import sinfof, squeuef, sacctf, sinfof_local, squeuef_local, sacctf_local, job_smi, expand_nodelist
from .screen import Display

def cache_result(ttl_seconds=30):
    """Cache decorator with time-based expiration."""
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()

            # Check if cached result exists and is still valid
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result

        return wrapper
    return decorator

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
    @cache_result(ttl_seconds=15)  # Cache for 15 seconds
    def squeue(self):
        if not hasattr(self, "_squeue"):
            if self.local:
                self._squeue = squeuef_local(self.clusters)
            else:
                self._squeue = squeuef(self.clusters)
        return self._squeue
    
    @property
    @cache_result(ttl_seconds=60)  # Cache for 60 seconds (sinfo changes less frequently)
    def sinfo(self):
        if not hasattr(self, "_sinfo"):
            if self.local:
                self._sinfo = sinfof_local(self.clusters)
            else:
                self._sinfo = sinfof(self.clusters)
        return self._sinfo
    
    @property
    def sacct(self):
        if not hasattr(self, "_sacct"):
            if self.local:
                self._sacct = sacctf_local(self.clusters)
            else:
                self._sacct = sacctf(self.clusters, self.start_time.strftime('%m%d%y'), self.end_time.strftime('%m%d%y'), partition=self.partition)
        return self._sacct

    @cache_result(ttl_seconds=300)  # Cache for 5 minutes (allocation strings are often repeated)
    def parse_alloc_string(self, string):
        """Optimized allocation string parser using single regex pass."""
        # Single regex to capture all values at once
        pattern = r'cpu=(\d+).*?mem=([^,]*).*?node=(\d+).*?gres/gpu=(\d+)'
        match = re.search(pattern, string, re.IGNORECASE)
        
        if match:
            cpus = int(match.group(1))
            mem_str = match.group(2)
            nodes = int(match.group(3))
            gpus = int(match.group(4))
        else:
            # Fallback to individual searches if combined pattern fails
            cpus = re.search(r'cpu=(\d+)', string, re.IGNORECASE)
            mem_str = re.search(r'(?<=mem=)[^,]*', string, re.IGNORECASE)
            nodes = re.search(r'node=(\d+)', string, re.IGNORECASE)
            gpus = re.search(r'gres/gpu=(\d+)', string, re.IGNORECASE)
            
            cpus = int(cpus.group(1)) if cpus else 0
            mem_str = mem_str.group(0) if mem_str else "0M"
            nodes = int(nodes.group(1)) if nodes else 0
            gpus = int(gpus.group(1)) if gpus else 0
        
        # Convert memory to megabytes
        n_mem = 0
        if mem_str.endswith("T"):
            n_mem = int(float(mem_str[:-1])) * 1024 * 1024 
        elif mem_str.endswith("G"):
            n_mem = int(float(mem_str[:-1])) * 1024
        elif mem_str.endswith("M"):
            n_mem = int(float(mem_str[:-1]))
        
        return {"cpus": cpus, "gpus": gpus, "mem": n_mem, "nodes": nodes}


    def process_jobs(self):
        """Process jobs from squeue output with optimized data structure operations.

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
            cluster_users = self.users.setdefault(cluster, OrderedDict())

            for job in self.squeue[cluster]['jobs']:
                user = job['user_name']
                user_data = cluster_users.setdefault(user, OrderedDict())

                group_name = job['group_name']
                if 'group' not in user_data:
                    user_data['group'] = group_name
                    user_data['branch'] = group_name.split("_")[0].upper()

                job_state = job['job_state']
                # tres_alloc_str is populated for running jobs; use it directly
                tres_str = job['tres_alloc_str'] if job_state == "RUNNING" and job['tres_alloc_str'] else ''
                if not tres_str:
                    # For pending jobs, build a synthetic TRES string from available fields
                    tres_str = f"cpu={job['num_cpus']},node={job['num_nodes']}"
                    # Try to extract GPU count from gres field
                    gpu_count = 0
                    if job.get('gres'):
                        gpu_match = re.search(r':(\d+)', job['gres'])
                        if gpu_match:
                            gpu_count = int(gpu_match.group(1))
                    if gpu_count:
                        tres_str += f",gres/gpu={gpu_count}"

                alloc = self.parse_alloc_string(tres_str)

                if job_state == "RUNNING":
                    running_jobs = user_data.setdefault("RUNNING", [])
                    running_jobs.append(alloc)
                    self.process_running_job(job)
                elif job_state == "PENDING":
                    pending_jobs = user_data.setdefault("PENDING", [])
                    pending_jobs.append(alloc)
                else:
                    print(f"{job_state} not considered yet!")

    def process_gpu_usage(self, node_name, gpus_string, n_nodes, cluster, user):
        """GPU usage parser. Handles both IDX-style strings and count-only gres strings.

        IDX-style: "gpu:tesla_v100-sxm2-16gb:1(IDX:0)"
        Count-only: "gpu:tesla_v100-sxm2-16gb:2"
        """
        # Try IDX-style first: "name:count(IDX:...)"
        pattern = r'([^:]+):(\d+)\(([^)]+)\)'
        match = re.search(pattern, gpus_string)

        if match:
            gpu_name = match.group(1)
            n_gpus = int(match.group(2))
            gpu_idx_str = match.group(3)

            gpu_range = []
            for i in gpu_idx_str.split(','):
                i = i.lstrip("IDX:")
                if '-' in i:
                    start, end = map(int, i.split('-'))
                    gpu_range.extend(range(start, end + 1))
                else:
                    gpu_range.append(int(i))
        else:
            # Count-only: "gpu:name:count" — allocate sequential indices
            parts = gpus_string.split(':')
            if len(parts) >= 3:
                gpu_name = parts[1]
                n_gpus = int(parts[2])
            elif len(parts) == 2:
                gpu_name = parts[0]
                n_gpus = int(parts[1])
            else:
                return

            if node_name not in self.resource_gpu.get(cluster, {}):
                return
            # Find first available GPU indices for this node
            for gname in self.resource_gpu[cluster][node_name]:
                if gname == gpu_name or gpu_name in gname or gname in gpu_name:
                    gpu_name = gname
                    break
            gpu_arr = self.resource_gpu[cluster][node_name].get(gpu_name)
            if gpu_arr is None:
                return
            available = np.where(gpu_arr == 0)[0][:n_gpus]
            gpu_range = list(available)

        is_single = (len(gpu_range) == 1) and (n_nodes <= 1)
        user_prefix = "S" if is_single else "P"

        self.resource_gpu[cluster][node_name][gpu_name][gpu_range] += 1
        self.resource_gpu_desc[cluster][node_name][gpu_name][gpu_range] = f"{user_prefix}{user}"

    def process_cpu_usage(self, node_name, cpus_on_node, n_nodes, cluster, user):
        """Sequential core fill: assign the next available core indices for this job."""
        if node_name not in self.resource_list.get(cluster, {}):
            return
        cores_array = self.resource_list[cluster][node_name]
        desc_array = self.resource_desc[cluster][node_name]

        available = np.where(cores_array == 0)[0][:cpus_on_node]
        cores_array[available] += 1

        if (cpus_on_node <= 1) and (n_nodes <= 1):
            desc_array[available] = f"S{user}"
        else:
            desc_array[available] = f"P{user}"

    def process_info(self):
        """ Node name,
            cores per socket,
            num sockets,
            gpus (gres total),
            node state,
            memory total,
            cpus used,
            cpus idle,
            core count,
        """
        for cluster, sinfo_list in self.sinfo.items():
            for info in sinfo_list['sinfo']:
                node_name = info['node_name']
                cores_per_socket = info['cores_per_socket']
                n_sockets = info['sockets']
                n_cpus = int(cores_per_socket) * int(n_sockets)
                gpus_string = info['gres_total']
                gpu_name = None
                gpu_count = 0
                node_states = info['node_states']
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
                    self.resource_list[cluster][node_name].fill(-1)
                    self.resource_desc[cluster][node_name].fill(-1)

                self.node_data.setdefault(cluster, OrderedDict())
                self.node_data[cluster][node_name] = {
                        'memory': info['memory_total'],
                        'memory_used': info['memory_allocated'],
                        'cpu_count': n_cpus,
                        'cpu_used': info['cpus_allocated'],
                        'cpu_idle': info['cpus_idle'],
                        'core_count': cores_per_socket,
                        'gpu_name': gpu_name,
                        'gpu_count': gpu_count,
                }

    def process_running_job(self, job):
        user = job['user_name']
        cluster = job['cluster']
        node_list_str = job.get('node_list', '')
        if not node_list_str:
            return

        node_names = expand_nodelist(node_list_str)
        if not node_names:
            return

        num_cpus = job['num_cpus']
        n_nodes = len(node_names)

        # Distribute CPUs across nodes (approximate even distribution)
        base_cpus = num_cpus // n_nodes if n_nodes > 0 else num_cpus
        remainder = num_cpus % n_nodes if n_nodes > 0 else 0

        for idx, node_name in enumerate(node_names):
            cpus_on_node = base_cpus + (1 if idx < remainder else 0)
            self.process_cpu_usage(node_name, cpus_on_node, n_nodes, cluster, user)

        # GPU allocation from gres field
        gres = job.get('gres', '')
        if gres:
            # gres can be like "gpu:tesla_v100-sxm2-16gb:1(IDX:0)" or "gpu:a100:2"
            # For multi-node, distribute GPU allocation across nodes
            for node_name in node_names:
                if node_name in self.resource_gpu.get(cluster, {}):
                    self.process_gpu_usage(node_name, gres, n_nodes, cluster, user)

    def gpu_report(self):
        """Prepare a usage report on GPUs.
        """
        # fill "" with NaN
        df = self.sacct.ffill()
        "gres/gpu:(.*?)="
        df['gpu_type'] = df['AllocTRES'].str.extract("gres/gpu:(.*?)=")
        df['gpu_count'] = df['AllocTRES'].str.extract("gres/gpu=(.*?),")
        df['gpu_count'] = pd.to_numeric(df['gpu_count'], errors='coerce')
        df['agency'] = df['Group'].str.split("_").str[0]
        # some entries in Group are blank so get agency from 'Account'?
        df['agency'] = df.apply(lambda x: x['agency'] if x['agency'].strip() != '' else x['Account'].split("_")[0], axis=1)
        
        df['agency'] = df['agency'].str.strip()
        df['agency'] = df['agency'].str.upper()

        #print(df.columns)
        if self.start_time.year != self.end_time.year:
            start_str = self.start_time.strftime('%a%e %B %Y')
        else:
            start_str = self.start_time.strftime('%a%e %B')
        end_str = self.end_time.strftime('%a%e %B %Y')
        TITLE = f"{self.clusters} GPU Count Per Job ({start_str} - {end_str})"
        dfgpu = df[~df['gpu_type'].isna()]
        # remove me
        #print(dfgpu.loc[dfgpu['gpu_count'] > 10, 'User'])
        #plt.hist(dfgpu['gpu_count'], alpha=0.5)
        ax = sns.histplot(data=dfgpu, x='gpu_count', hue='agency', multiple='stack')
        ax.set_yscale('log')
        ax.set_xlabel("GPUs Used")
        ax.set_ylabel("Number of Jobs")
        plt.title(TITLE)
        plt.savefig(self.output)
        print(f"File saved to {self.output}.")

    def __call__(self):
        """Calling the cluster stat instance because I couldn't think of a good name for the function other 
        than `cluster_stat` which is redundant."""

        self.process_info() # must be called first
        self.process_jobs() # second..

def valid_datetime(arg_str):
    try:
        return datetime.strptime(arg_str, '%m%d%y')
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format '{arg_str}'. Expected format: mmddyy (e.g. Oct 7, 2025 == '100725')"
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Graphical representation of HPC usage.")
    cluster_group = parser.add_argument_group("Cluster Stat Display")
    report_group = parser.add_argument_group("Cluster Usage Report")
    graphic_group = parser.add_argument_group("GPU Running Usage Report")
    parser.add_argument('--clusters', '-M', default='all', help='Specify the cluster to display on screen.')
    parser.add_argument('--partition', '-P', default='', help='Specify a particular partition to extract data from.')
    parser.add_argument('--local', action='store_true', help='Tell the program to search the directory for GPU usage files.')
    cluster_group.add_argument('--gpus-only', '-g', action='store_true', help='Print out only the GPU nodes.')
    report_group.add_argument('--start-time', '-S', action='store', 
                              help='Specify the start time for the report generation. Default is one month back.',
                              type=valid_datetime,
                              default=(datetime.now() - relativedelta(months=1)))
    report_group.add_argument('--end-time', '-E', action='store', 
                              help='Specify the end time for the report generation. Default is today.',
                              type=valid_datetime,
                              default=datetime.now())
    report_group.add_argument('--output', '-o', action='store', 
                              help='Set the output file name for the report (plot).',
                              default=os.path.join(os.getcwd(), "plot.png"))
    graphic_group.add_argument('jobid', nargs='?', type=int, help='Specify the SLURM jobid for the GPUs you wish to see.')

    return parser.parse_args()

def job_main():
    args = parse_args()
    args_dict = vars(args)
    if 'jobid' not in args_dict:
        print(f"Error: please specify a jobid as a positional argument to the command.")
        sys.exit()
    screen = Display(**args_dict)
    screen.print_gpu_usage()

def sacct_main():
    args = parse_args()
    args_dict = vars(args)
    cs = ClusterStat(**args_dict)
    cs.gpu_report()
    
def main():
    args = parse_args()
    args_dict = vars(args)
    cs = ClusterStat(**args_dict)
    cs()
    screen = Display(**args_dict)
    screen.cluster_stat = cs
    screen.initialize_usercodes()
    print(screen)

if __name__ == '__main__':
    main()

