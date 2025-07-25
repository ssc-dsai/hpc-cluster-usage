import numpy as np
import xmltodict
import os
from math import ceil
import sys
import getpass
import pyfiglet
from collections import OrderedDict
import json
from .sqstat import job_smi
import re
np.set_printoptions(linewidth=np.inf, threshold=np.inf)

def parse_gpu_info(gpu_data):
    """ memory reported in MiB.
    1 GB == 953.674 MiB

    util's reported in %
    """
    total_mem = re.search(r'(\d+) MiB', gpu_data['fb_memory_usage']['total']).group(1)
    free_mem = re.search(r'(\d+) MiB', gpu_data['fb_memory_usage']['free']).group(1)
    used_mem = re.search(r'(\d+) MiB', gpu_data['fb_memory_usage']['used']).group(1)
    res_mem = re.search(r'(\d+) MiB', gpu_data['fb_memory_usage']['reserved']).group(1) 
    name = gpu_data['product_name']
    ide = gpu_data['@id']
    # below utils will not be reported if MIG is enabled - Multi-Instance GPU: partitioning a GPU into smaller GPUs.
    core_util = re.search(r'(\d+) %', gpu_data['utilization']['gpu_util']).group(1)
    mem_util = re.search(r'(\d+) %', gpu_data['utilization']['memory_util']).group(1)
    # decoder_util, encoder_util, jpeg_util, ofa_util
    return {"name": name,
            "id": ide,
            "core_util": core_util,
            "mem_util": mem_util,
            "total_mem": total_mem,
            "free_mem": free_mem,
            "used_mem": used_mem,
            "res_mem": res_mem,
    }

def replace_with_ranges(string):
    numbers = sorted(map(int, string.split(",")))
    ranges = []
    start = numbers[0]
    end = numbers[0]
    for num in numbers[1:]:
        if num == end+1:
            end = num
        else:
            if (start == end):
                ranges.append(f"{start}")
            elif start == (end-1):
                ranges.append(f"{start}")
                ranges.append(f"{end}")
            else:
                ranges.append(f"{start}-{end}")
            start = num
            end = num
    if start == end:
        ranges.append(f"{start}")
    elif start == (end-1):
        ranges.append(f"{start}")
        ranges.append(f"{end}")
    else:
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)

class Display:
    """Display the cluster usage to the screen."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) # just put em all as attributes why not?
        try:
            self._width, self._height = os.get_terminal_size()
        except OSError:
            self._width, self._height = 100, 70
        # index errors :(
        self._width, self._height = (2000, 900)
        # maximum length a node can be represented on a terminal line
        self.max_repr_len = 64
        self.max_line_length = 0 # currently assume 2 columns worth of data to the screen
        # line_length determines where the second column starts.
        self.row = 0
        self._screen = np.empty((self._height, self._width), dtype='U36')
        self._screen[:] = " "
        self._usercodes = OrderedDict()
        self._cluster_stat = None

    @property
    def flush_screen(self):
        del self._screen
        self._screen = np.empty((self._height, self._width), dtype='U36')
        self._screen[:] = " "

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

        for cluster in self.cluster_stat.users:
            for user_id, user in enumerate(self.cluster_stat.users[cluster], start=1):
                try:
                    self._usercodes[user]
                except KeyError:
                    user_id = user_id%256
                    # avoid black and darker blues 
                    user_id = user_id+1 if user_id >= 16 else user_id
                    user_id = user_id+1 if user_id >= 17 else user_id
                    user_id = user_id+1 if user_id >= 18 else user_id
                    user_id = user_id+1 if user_id >= 19 else user_id
                    user_id = user_id+1 if user_id >= 20 else user_id
                    user_id = user_id+1 if user_id >= 21 else user_id
                    #print(f"{user=}: {user_id=}")
                    #print(f"{user=}: {user_id=}")
                    self._usercodes[user] = f"\033[38;5;{user_id}m"
        # keep the active user the same color
        self._usercodes[getpass.getuser()] = "\033[104;39;1m"

    def cpu_data_to_screen(self, cluster, row, column):
        # now do the nodes
        self.row = row
        for node in sorted(self.cluster_stat.resource_list[cluster]):
            desc = self.cluster_stat.resource_desc[cluster][node]
            res = self.cluster_stat.resource_list[cluster][node]
            not_used = np.where(res == 0)
            used = np.where(res > 0)
            bad = np.where(res == -1)
            #n_used += used[0].shape[0]  
            #n_available += not_used[0].shape[0]
            #n_total += (used[0].shape[0] + not_used[0].shape[0])
            # set cpus not used to green
            desc[not_used] = '\033[0m\033[42m \033[0m'
            desc[bad] = '\033[0m\033[41m\033[5m\033[1;37mX\033[0m'
            for i, idx in np.ndenumerate(used):
                e = desc[idx]
                user=e[1:].strip()
                s_or_p = "=" if e[0] == "P" else "+"
                desc[idx] = f"{self._usercodes[user]}{s_or_p}\033[0m"

            # if the number of cores is too long for the screen, split the line
            if desc.shape[0] > self.max_repr_len:
                # indent for the padding of the sencond+ row(s)
                indent = 6
                desc_split = np.array_split(desc, ceil(desc.shape[0]/self.max_repr_len))
                for idx, sub_desc in enumerate(desc_split):
                    # first row
                    if idx == 0:
                        line = (list(f"{node:>13s}") + 
                                [":", "["] + 
                                sub_desc.tolist() 
                        )

                        line[-1] += "\033[0m"
                        line = np.array(line, dtype='U36')
                        self._screen[row, column: column + len(line)] = line
                    # middle rows
                    elif idx < (len(desc_split) - 1):
                        line = ([" "]*indent) + sub_desc.tolist()
                        line[-1] += "\033[0m"
                        line = np.array(line, dtype='U36')
                        self._screen[row, column: column + len(line)] = line
                        # have to account for empty characters (there are 3 in the first row)
                        #self._screen[row, column+len(line): column+len(line)+3] = ''
                    # terminus
                    else:
                        line = ([" "]*indent) + sub_desc.tolist() + ["]"]
                        line = np.array(line, dtype='U36')
                        self._screen[row, column: column + len(line)] = line
                        # have to account for empty characters (there are 3 in the first row)
                        #self._screen[row, column+len(line): column+len(line)+3] = ''
                    self.max_line_length = max(len(line), self.max_line_length)
                    row += 1

            else:
                line = list(f"{node:>13s}") + [":", "["] + desc.tolist() 
                line[-1] += "\033[0m]"
                line = np.array(line, dtype='U36')
                self._screen[row, column: column + len(line)] = line
                # nb the bash color script codes above do not add to the final output column count.
                self.max_line_length = max(len(line), self.max_line_length)
                row += 1
        self.row = row

    def summary_to_screen(self, cluster, row, column):
        self.row = row
        n_used, n_available, n_total = 0, 0, 0
        for node in self.cluster_stat.node_data[cluster]:
            node_data = self.cluster_stat.node_data[cluster][node]
            # this doesn't seem to reflect user data. so not sure what's happening here.
            n_total += node_data['cpu_count']
            n_used += node_data['cpu_used']
            n_available += node_data['cpu_idle']

        nline = list(f"{'used:':>12s}{n_used:5d}")
        self._screen[row, column: column+len(nline)] = nline
        row += 1
        nline = list(f"{'available:':>12s}{n_available:5d}")
        self._screen[row, column: column+len(nline)] = nline
        row += 1
        nline = list(f"{'total:':>12s}{n_total:5d}")
        self._screen[row, column: column+len(nline)] = nline
        row += 2
        self.row = row
   
    def user_data_to_screen(self, cluster, row, column):
        self.row = row
        max_user_length = max(len(x) for x in self.cluster_stat.users[cluster]) + 2

        max_branch_length = max(len(val['branch']) for x,val in self.cluster_stat.users[cluster].items()) + 2

        h = list([f"{'USER':>{max_user_length}s}",
                  f"{'AGENCY':>{max_branch_length}s}",
                  f"{'CPU_R':>8s}", 
                  f"{'CPU_Q':>8s}", 
                  f"{'GPU_R':>8s}", 
                  f"{'GPU_Q':>8s}"])
        nline = np.array(h, dtype='U36')
        self._screen[row, column:column+len(nline)] = nline
        row += 1
        # 
        # user stats
        #
        for user in sorted(self.cluster_stat.users[cluster]):
            stats = self.cluster_stat.users[cluster][user]
            cpu_r, cpu_q, gpu_r, gpu_q, mem_r, mem_q = 0, 0, 0, 0, 0, 0
            if 'RUNNING' in stats.keys():
                cpu_r = sum([i['cpus'] for i in stats['RUNNING']])
                gpu_r = sum([i['gpus'] for i in stats['RUNNING']])
                mem_r = sum([i['mem'] for i in stats['RUNNING']]) # megabytes
            if 'PENDING' in stats.keys():
                cpu_q = sum([i['cpus'] for i in stats['PENDING']])
                gpu_q = sum([i['gpus'] for i in stats['PENDING']])
                mem_q = sum([i['mem'] for i in stats['PENDING']]) # megabytes

            nline = [f"{self._usercodes[user]}"]
            nline += list([f"{user:>{max_user_length}s}", 
                           f"{stats['branch']:>{max_branch_length}s}",
                           f"{cpu_r:>8d}", 
                           f"{cpu_q:>8d}",
                           f"{gpu_r:>8d}",
                           f"{gpu_q:>8d}",
                     ])
            nline[-1] += "\033[0m"
            nline = np.array(nline, dtype='U36')
            if self.gpus_only and ((gpu_q > 0) or (gpu_r > 0)):
                self._screen[row, column: column+len(nline)] = nline
                row += 1
            elif not self.gpus_only:
                self._screen[row, column: column+len(nline)] = nline
                row += 1
        self.row = row

    def gpu_data_to_screen(self, cluster, row, column):
        if cluster not in self.cluster_stat.resource_gpu.keys():
            return
        for node in self.cluster_stat.resource_gpu[cluster]:
            for gpu_type in self.cluster_stat.resource_gpu[cluster][node]:
                res = self.cluster_stat.resource_gpu[cluster][node][gpu_type]
                desc = self.cluster_stat.resource_gpu_desc[cluster][node][gpu_type]
                not_used = np.where(res == 0)
                used = np.where(res > 0)
                bad = np.where(res == -1)
                desc[not_used] = '\033[0m\033[42m \033[0m'
                desc[bad] = '\033[0m\033[41m\033[5m\033[1;37mX\033[0m'
                for i, idx in np.ndenumerate(used):
                    e = desc[idx]
                    user=e[1:].strip()
                    s_or_p = "=" if e[0] == "P" else "+"
                    desc[idx] = f"{self._usercodes[user]}{s_or_p}\033[0m"

                line =  list(f"{node:>10s}") + [" ", "("] + list(gpu_type) + [")", ":", "["] + desc.tolist() + ["\033[0m]"]
                line = np.array(line, dtype='U36')
                self._screen[row, column: column + len(line)] = line

                self.max_line_length = max(len(line), self.max_line_length)
                # nb the bash color script codes above do not add to the final output column count.
                row += 1
        self.row = row

    def gpu_title(self, cluster, row, column):
        if cluster not in self.cluster_stat.resource_gpu.keys():
            return
        self.row = row
        title = pyfiglet.figlet_format("GPU Nodes", justify='left', font='small').split("\n")
        title = [list(i) for i in title]
        max_title_len = max(len(t) for t in title)
        # prepare the multi-line title :)
        for t in title:
            if t:
                t[0] = "\033[5m" + t[0]
                t[-1] = t[-1] + "\033[0m"
                self._screen[row, column:column+len(t)] = t 
            row+=1
        self.row = row

    def gpu_progress_bar(self, value, maximum, width=12, show_value=True, colour='\033[0;34m',
            label=None, append=""):
        """
        Generate a visual representation of progress.
        """
        bar = []
        if label is not None:
            bar.append(label)
   
        used = int(width*value/maximum)
        if used > maximum:
            used = width
        elif used < 0:
            used = 0
        unused = width - used
        bar.extend(['[', colour, '|'*used, '.'*unused, '\033[0m', ']'])
   
        if show_value:
            bar.append(f' {int(value)}/{int(maximum)}')
            bar.append(f' {append}')
   
        return "".join(bar)

    def gpu_summary(self, cluster, row, column):
        self.row = row
        min_gpu = np.inf
        max_gpu, free_nodes, total_free = 0, 0, 0
        if cluster not in self.cluster_stat.resource_gpu.keys():
            return
        for node in self.cluster_stat.resource_gpu[cluster]:
            for gpu_name in self.cluster_stat.resource_gpu[cluster][node]:
                gpu_arr = self.cluster_stat.resource_gpu[cluster][node][gpu_name]
                empty_idx = np.where(gpu_arr == 0)
                n_empty = empty_idx[0].shape[0]
                if n_empty != 0:
                    if (free_nodes == 0):
                        line = list("Unallocated GPUs:")
                        self._screen[row, column: column + len(line)] = line
                        row += 1
                    free_nodes += 1
                    range_string = ", ".join([str(i) for i in empty_idx[0]])
                    truncated_string = replace_with_ranges(range_string)
                    line = list(f"{node}:{gpu_name}:{n_empty}(IDX:{truncated_string})")
                    self.max_line_length = max(len(line), self.max_line_length)
                    self._screen[row, column: column+len(line)] = line
                    row += 1
                    max_gpu = n_empty if n_empty > max_gpu else max_gpu
                    min_gpu = n_empty if n_empty < min_gpu else min_gpu
                    total_free += n_empty
        if free_nodes != 0:
            line = list(f"Total GPUs available: {total_free} across {free_nodes} nodes.")
            self.max_line_length = max(len(line), self.max_line_length)
            self._screen[row, column:column+len(line)] = line
            row += 1
            line = list(f"Per node: (MIN {min_gpu}, MAX {max_gpu})")
            self.max_line_length = max(len(line), self.max_line_length)
            self._screen[row, column:column+len(line)] = line
            row += 1

    def print_gpu_usage(self):
        DIV=1000 #953.674
        # below has a hard time when multi node configuration of job.
        # google "slurm how to log into a job" for multi-node suggestions..
        # https://stackoverflow.com/questions/63366098/rejoin-a-bash-slurm-job
        # Try the 'ssh=True' option in the comments section of the job submission.
        # https://portal.science.gc.ca/xwiki/bin/view/Projects/Science/Tutorials%20and%20HowTos/Quick%20Start%20to%20Using%20Linux%20Clusters%20With%20SLURM/
        out = job_smi(self.jobid, cluster=self.cluster)

        #out = new_job_smi_output()
        out_spl = [i for i in out.split("NODENAME=") if i]
        # something wierd going on with multi nodes.
        for node in out_spl:
            out_lines = node.splitlines()
            (node_name, procid) = out_lines[0].split(":")
            data = xmltodict.parse("\n".join(out_lines[1:]))
            title = pyfiglet.figlet_format(node_name, justify='center', font='small')
            print(title)
            #print(json.dumps(data,
            #                 sort_keys=True,
            #                 indent=4,
            #                 separators=(',', ': '),
            #     )
            #)
            gpus_on_node = int(data['nvidia_smi_log']['attached_gpus'])
            if gpus_on_node == 1:
                gpu = parse_gpu_info(data['nvidia_smi_log']['gpu'])
                print(f"{gpu['name']}: {gpu['id']}")
                mem_bar = self.gpu_progress_bar(float(gpu['used_mem'])/DIV, 
                                                float(gpu['total_mem'])/DIV, 
                                                width=40,
                                                label=f"{'Memory':>20s}",
                                                append="GB")
                gpu_bar = self.gpu_progress_bar(int(gpu['core_util']),
                                                100,
                                                width=40,
                                                label=f"{'GPU Core Usage':>20s}",
                                                colour='\033[0;32m',
                                                append="%") 
        
                #print(f"{gpu['name']}:{gpu['id']}:{gpu['core_util']}:{gpu['mem_util']}:{gpu['total_mem']}")
                print(gpu_bar)
                print(mem_bar)
                print()
            else:
                for gpu_data in data['nvidia_smi_log']['gpu']:
                    gpu = parse_gpu_info(gpu_data)
                    #print(f"{gpu['name']}:{gpu['id']}:{gpu['core_util']}:{gpu['mem_util']}:{gpu['total_mem']}")
                    print(f"{gpu['name']}: {gpu['id']}")
                    mem_bar = self.gpu_progress_bar(float(gpu['used_mem'])/DIV, 
                                                    float(gpu['total_mem'])/DIV, 
                                                    width=40,
                                                    label=f"{'Memory':>20s}",
                                                    append="GB",
                    )
                    gpu_bar = self.gpu_progress_bar(int(gpu['core_util']),
                                                    100,
                                                    width=40,
                                                    label=f"{'GPU Core Usage':>20s}",
                                                    colour='\033[0;32m',
                                                    append="%",
                    )
                    print(gpu_bar)
                    print(mem_bar)
                    print()

    def cluster_title(self, cluster, row, column):
        title = pyfiglet.figlet_format(cluster, justify='center', font='standard').split("\n")
        title = [list(i) for i in title]

        max_title_len = max(len(t) for t in title)
        # prepare the multi-line title :)
        for t in title:
            if t:
                t[0] = "\033[5m" + t[0]
                t[-1] = t[-1] + "\033[0m"
                self._screen[row, column:column+len(t)] = t 
            row+=1
        self.row = row + 2

    def format_output(self, cluster):
        """Prepare the _screen array with colors and such."""
        self.flush_screen
        if self.cluster_stat is None:
            print(f"Warning: format_output was called but cluster_stat was not defined!")
            return
        
        if self.gpus_only and (cluster in self.cluster_stat.resource_gpu.keys()):

            self.cluster_title(cluster, 0, 0)
            
            default_row = self.row
            
            self.gpu_data_to_screen(cluster, default_row, 0)
            
            self.gpu_summary(cluster, self.row+2, 0)
             
            self.user_data_to_screen(cluster, default_row, self.max_line_length+4)


        elif not self.gpus_only:
            self.cluster_title(cluster, 0, 0)
            
            default_row = self.row
            
            self.cpu_data_to_screen(cluster, default_row, 0)

            self.summary_to_screen(cluster, default_row, self.max_line_length+4)
        
            self.user_data_to_screen(cluster, self.row, self.max_line_length+4)

            self.gpu_title(cluster, self.row+2, self.max_line_length+4)
            max_length = self.max_line_length

            self.gpu_data_to_screen(cluster, self.row, max_length+4)
        
            self.gpu_summary(cluster, self.row+2, max_length+4)

        lines = ("".join(x for x in line).rstrip() for line in self._screen)
        text = "\n".join(line for line in lines if line.strip())
        return (text)

    def __repr__(self):
        return repr(self._screen)

    def __str__(self):
        all_strings = ""
        for cluster in self.cluster_stat.resource_list:
            all_strings += "\n"
            all_strings += (self.format_output(cluster))
        return all_strings
