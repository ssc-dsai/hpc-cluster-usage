import numpy as np
import os
import sys
import getpass
import pyfiglet
from collections import OrderedDict
np.set_printoptions(linewidth=np.inf, threshold=np.inf)

class Display:
    """Display the cluster usage to the screen."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) # just put em all as attributes why not?
        try:
            self._width, self._height = os.get_terminal_size()
        except OSError:
            self._width, self._height = 100, 70

        # index errors :(
        self._width, self._height = (2000, 200)
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

        for cluster in self.cluster_stat.users:
            for user_id, user in enumerate(self.cluster_stat.users[cluster], start=1):
                try:
                    self._usercodes[user]
                except KeyError:
                    user_id = user_id%15
                    self._usercodes[user] = f"\033[38;5;{user_id}m"
        # keep the active user the same color
        self._usercodes[getpass.getuser()] = "\033[104;39;1m"

    def format_output(self, cluster):
        """Prepare the _screen array with colors and such."""
        if self.cluster_stat is None:
            print(f"Warning: format_output was called but cluster_stat was not defined!")
            return

        # formatting is cluster-specific
        lcolumn = 0
        row = 0
        max_line_length = 0
        n_used, n_available, n_total = 0, 0, 0
        cluster_name = "\033[5m" + pyfiglet.figlet_format(cluster, justify='center', font='standard') + "\033[0m"
        print(cluster_name)
        for node in sorted(self.cluster_stat.resource_list[cluster]):
            desc = self.cluster_stat.resource_desc[cluster][node]
            res = self.cluster_stat.resource_list[cluster][node]
            not_used = np.where(res == 0)
            used = np.where(res != 0)
            n_used += used[0].shape[0]  
            n_available += not_used[0].shape[0]
            n_total += (used[0].shape[0] + not_used[0].shape[0])
            # set cpus not used to green
            desc[not_used] = '\033[0m\033[42m \033[0m'
            for i, idx in np.ndenumerate(used):
                e = desc[idx]
                user=e[1:].strip()
                s_or_p = "=" if e[0] == "P" else "+"
                desc[idx] = f"{self._usercodes[user]}{s_or_p}\033[0m"

            line = list(f"\033[1m{node:>13s}\033[0m:") +  ["["] + desc.tolist() + ["]\033[0m"]
            line = np.array(line, dtype='U36')
            if len(line) > self._screen.shape[1]:
                line = line[:self._screen.shape[1]]
            self._screen[row, lcolumn: lcolumn + len(line)] = line
            max_line_length = max(len(line), max_line_length)
            row += 1
        
        lcolumn += max_line_length + 4
        row = 0
        nline = list(f"{'used:':>12s}{n_used:5d}")
        self._screen[row, lcolumn: lcolumn+len(nline)] = nline
        row += 1
        nline = list(f"{'available:':>12s}{n_available:5d}")
        self._screen[row, lcolumn: lcolumn+len(nline)] = nline
        row += 1
        nline = list(f"{'total:':>12s}{n_total:5d}")
        self._screen[row, lcolumn: lcolumn+len(nline)] = nline
        row += 2

        max_user_length = max(len(x) for x in self.cluster_stat.users[cluster])

        h = list([f"{'USER':>{max_user_length}s}", 
                  f"{'CPU_R':>6s}", 
                  f"{'CPU_Q':>6s}", 
                  f"{'GPU_R':>6s}", 
                  f"{'GPU_Q':>6s}"])
        nline = np.array(h, dtype='U36')
        self._screen[row, lcolumn:lcolumn+len(nline)] = nline
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
                           f"{cpu_r:>6d}", 
                           f"{cpu_q:>6d}",
                           f"{gpu_r:>6d}",
                           f"{gpu_q:>6d}",
                     ])
            nline += ["\033[0m"]
            nline = np.array(nline, dtype='U36')
            self._screen[row, lcolumn: lcolumn+len(nline)] = nline
            row += 1

        lines = ("".join(x for x in line).rstrip() for line in self._screen)
        text = "\n".join(line for line in lines if line.strip())
        return (text)

    def __repr__(self):
        return repr(self._screen)

    def __str__(self):
        all_strings = ""
        for cluster in self.cluster_stat.resource_list:
            all_strings += (self.format_output(cluster))
        return all_strings
