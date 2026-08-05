import numpy as np
import xmltodict
import os
from math import ceil
import sys
import getpass
import pyfiglet
from collections import OrderedDict
import json
import time
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

    # How an unallocated slot is drawn, keyed by node health. Anything other
    # than "ok" means the slot is not really available to a new job.
    FREE_CELL = {
        "ok":       '\033[0m\033[42m \033[0m',                  # green   - free
        "mem":      '\033[0m\033[45m\033[1;37mm\033[0m',        # magenta - no memory left
        "drain":    '\033[0m\033[43m\033[1;30md\033[0m',        # amber   - draining
        "reserved": '\033[0m\033[46m\033[1;30mr\033[0m',        # cyan    - reserved
        "down":     '\033[0m\033[41m\033[5m\033[1;37mX\033[0m', # red     - down
    }

    def _node_glyph(self, cluster, node):
        """Classify a node so free slots can be drawn honestly."""
        info = self.cluster_stat.node_data.get(cluster, {}).get(node)
        if info is None:
            return "ok"
        status = info.get('status', 'ok')
        if status != "ok":
            return status
        return "mem" if info.get('mem_blocked') else "ok"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) # just put em all as attributes why not?
        # The buffer starts small and grows on demand via _blit, so there is no
        # need to over-allocate to dodge index errors. Each cell is a U36 string
        # (144 bytes), so the old 900x2000 buffer was 259MB and cost ~0.9s per
        # cluster just to allocate and join -- of which under 2% was ever used.
        self._height, self._width = (256, 512)
        # maximum length a node can be represented on a terminal line
        self.max_repr_len = 64
        self.max_line_length = 0 # currently assume 2 columns worth of data to the screen
        # line_length determines where the second column starts.
        self.row = 0
        self._screen = np.full((self._height, self._width), " ", dtype='U36')
        # extents actually written, so rendering only joins live cells
        self._max_row = 0
        self._max_col = 0
        self._usercodes = OrderedDict()
        self._cluster_stat = None

    @property
    def flush_screen(self):
        # reuse the allocation; refilling is much cheaper than reallocating
        self._screen[:self._max_row, :self._max_col] = " "
        self._max_row = 0
        self._max_col = 0
        # this is per-cluster layout state; leaving it set made each successive
        # cluster indent its right-hand panel further across the screen
        self.max_line_length = 0

    def _grow(self, min_height, min_width):
        """Enlarge the buffer to fit at least (min_height, min_width)."""
        new_h = max(min_height, self._height)
        new_w = max(min_width, self._width)
        while self._height < new_h:
            self._height *= 2
        while self._width < new_w:
            self._width *= 2
        grown = np.full((self._height, self._width), " ", dtype='U36')
        grown[:self._max_row, :self._max_col] = self._screen[:self._max_row, :self._max_col]
        self._screen = grown

    def _blit(self, row, column, cells):
        """Write a row of cells into the buffer, growing it if needed.

        Tracks the written extents so format_output can join only the region
        that actually holds output.
        """
        if not len(cells):
            return
        cells = np.asarray(cells, dtype='U36')
        end = column + cells.shape[0]
        if (row >= self._height) or (end > self._width):
            self._grow(row + 1, end)
        self._screen[row, column:end] = cells
        if (row + 1) > self._max_row:
            self._max_row = row + 1
        if end > self._max_col:
            self._max_col = end

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
            # copy: the loop below replaces "Pjsmith" style entries with the
            # escape sequences that draw them, so rendering in place would
            # destroy the source data and break any second pass over it
            desc = self.cluster_stat.resource_desc[cluster][node].copy()
            res = self.cluster_stat.resource_list[cluster][node]
            not_used = np.where(res == 0)
            used = np.where(res > 0)
            #n_used += used[0].shape[0]
            #n_available += not_used[0].shape[0]
            #n_total += (used[0].shape[0] + not_used[0].shape[0])
            # unallocated cells take their colour from the node's health, so a
            # slot that cannot actually accept work never shows up as green
            desc[not_used] = self.FREE_CELL[self._node_glyph(cluster, node)]
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
                        self._blit(row, column, line)
                    # middle rows
                    elif idx < (len(desc_split) - 1):
                        line = ([" "]*indent) + sub_desc.tolist()
                        line[-1] += "\033[0m"
                        line = np.array(line, dtype='U36')
                        self._blit(row, column, line)
                        # have to account for empty characters (there are 3 in the first row)
                        #self._screen[row, column+len(line): column+len(line)+3] = ''
                    # terminus
                    else:
                        line = ([" "]*indent) + sub_desc.tolist() + ["]"]
                        line = np.array(line, dtype='U36')
                        self._blit(row, column, line)
                        # have to account for empty characters (there are 3 in the first row)
                        #self._screen[row, column+len(line): column+len(line)+3] = ''
                    self.max_line_length = max(len(line), self.max_line_length)
                    row += 1

            else:
                line = list(f"{node:>13s}") + [":", "["] + desc.tolist() 
                line[-1] += "\033[0m]"
                line = np.array(line, dtype='U36')
                self._blit(row, column, line)
                # nb the bash color script codes above do not add to the final output column count.
                self.max_line_length = max(len(line), self.max_line_length)
                row += 1
        self.row = row

    def summary_to_screen(self, cluster, row, column):
        self.row = row
        n_used, n_available, n_total = 0, 0, 0
        m_used, m_total = 0, 0
        unreachable = 0
        health = OrderedDict()
        for node in self.cluster_stat.node_data[cluster]:
            node_data = self.cluster_stat.node_data[cluster][node]
            # this doesn't seem to reflect user data. so not sure what's happening here.
            n_total += node_data['cpu_count']
            n_used += node_data['cpu_used']
            m_total += node_data['memory']
            m_used += node_data['memory_used']
            glyph = self._node_glyph(cluster, node)
            if glyph == "ok":
                n_available += node_data['cpu_idle']
            else:
                # idle cores here cannot take new work
                unreachable += node_data['cpu_idle']
                health[glyph] = health.get(glyph, 0) + 1

        def put(text, r):
            # NB deliberately does not touch max_line_length: that tracks the
            # width of the left-hand node column, which is what positions this
            # panel. Feeding it back in would push the panel right each cluster.
            self._blit(r, column, list(text))
            return r + 1

        row = put(f"{'used:':>12s}{n_used:5d}", row)
        row = put(f"{'available:':>12s}{n_available:5d}", row)
        row = put(f"{'total:':>12s}{n_total:5d}", row)
        if unreachable:
            detail = ", ".join(f"{v} {k}" for k, v in health.items())
            row = put(f"{'unreachable:':>12s}{unreachable:5d}  ({detail})", row)
        if m_total:
            row = put(f"{'memory:':>12s}{m_used/1048576:5.1f}/{m_total/1048576:.1f} TB"
                      f" ({100*m_used/m_total:.0f}%)", row)

        # when capacity frees up, from the running jobs' end times
        freeing = self._freeing_soon(cluster)
        if freeing is not None:
            cpu_1h, gpu_1h, n_1h = freeing
            row = put(f"{'freeing 1h:':>12s}{cpu_1h:5d} CPU, {gpu_1h} GPU ({n_1h} jobs)", row)

        # next scheduled maintenance window
        for text in self._downtime_lines(cluster):
            row = put(text, row)

        # pending jobs that will never run without intervention
        reasons = self.cluster_stat.pending_reasons.get(cluster)
        if reasons:
            blocked = {k: v for k, v in reasons.items()
                       if k in self.cluster_stat.DEAD_PENDING_REASONS}
            waiting = sum(v for k, v in reasons.items() if k not in blocked)
            row += 1
            row = put(f"queued: {waiting} runnable, {sum(blocked.values())} blocked", row)
            for reason, count in sorted(blocked.items(), key=lambda kv: -kv[1])[:3]:
                row = put(f"   \033[2m{count:>5d} {reason}\033[0m", row)
        row += 1
        self.row = row

    @staticmethod
    def _duration(seconds):
        """Compact d/h/m rendering for a span of time."""
        seconds = int(max(0, seconds))
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60
        if days:
            return f"{days}d{hours:02d}h"
        if hours:
            return f"{hours}h{minutes:02d}m"
        return f"{minutes}m"

    def _downtime_lines(self, cluster):
        """Render the next maintenance window, loudest when it is imminent."""
        dt = self.cluster_stat.downtime.get(cluster)
        if dt is None:
            return []

        start = time.strftime('%a %d %b %H:%M', time.localtime(dt['start']))
        window = self._duration(dt['end'] - dt['start'])
        scope = "whole cluster" if dt['whole_cluster'] else f"{dt['node_count']} nodes"

        if dt['active']:
            ends = time.strftime('%a %d %b %H:%M', time.localtime(dt['end']))
            return [f"\033[1;41;97m DOWNTIME NOW \033[0m until {ends} ({scope})"]

        lead = dt['max_walltime']
        # highlight once the window is close enough to affect what you submit
        colour = "\033[1;33m" if lead < 7 * 86400 else "\033[2m"
        lines = [f"{colour}downtime: {start} +{window} ({scope})\033[0m"]
        lines.append(f"{colour}   max walltime until then: {self._duration(lead)}\033[0m")
        return lines

    def _freeing_soon(self, cluster, horizon=3600):
        """CPU/GPU due to be released by running jobs within `horizon` seconds."""
        now = time.time()
        cpu, gpu, jobs = 0, 0, 0
        seen = False
        for user, stats in self.cluster_stat.users.get(cluster, {}).items():
            for job in stats.get('RUNNING', []):
                end = job.get('end_time')
                if not end:
                    continue
                seen = True
                if now < end <= (now + horizon):
                    cpu += job['cpus']
                    gpu += job['gpus']
                    jobs += 1
        return (cpu, gpu, jobs) if seen else None
   
    def user_data_to_screen(self, cluster, row, column):
        self.row = row
        # keep at least one space of padding around the headings themselves,
        # otherwise short names let "USER" and "AGENCY" run together
        max_user_length = max(max(len(x) for x in self.cluster_stat.users[cluster]) + 2,
                              len('USER') + 1)

        max_branch_length = max(max(len(val['branch']) for x,val in self.cluster_stat.users[cluster].items()) + 2,
                                len('AGENCY') + 1)

        h = list([f"{'USER':>{max_user_length}s}",
                  f"{'AGENCY':>{max_branch_length}s}",
                  f"{'CPU_R':>8s}",
                  f"{'CPU_Q':>8s}",
                  f"{'GPU_R':>8s}",
                  f"{'GPU_Q':>8s}",
                  f"{'MEM_R/GB':>10s}",
                  f"{'MEM_Q/GB':>10s}",
                  f"{'STUCK':>7s}"])
        nline = np.array(h, dtype='U36')
        self._blit(row, column, nline)
        row += 1
        #
        # user stats
        #
        for user in sorted(self.cluster_stat.users[cluster]):
            stats = self.cluster_stat.users[cluster][user]
            cpu_r, cpu_q, gpu_r, gpu_q, mem_r, mem_q = 0, 0, 0, 0, 0, 0
            stuck = 0
            if 'RUNNING' in stats.keys():
                cpu_r = sum([i['cpus'] for i in stats['RUNNING']])
                gpu_r = sum([i['gpus'] for i in stats['RUNNING']])
                mem_r = sum([i['mem'] for i in stats['RUNNING']]) # megabytes
            if 'PENDING' in stats.keys():
                # queued totals count only jobs that can actually still run;
                # blocked ones are reported separately under STUCK
                live = [i for i in stats['PENDING'] if not i.get('blocked')]
                stuck = len(stats['PENDING']) - len(live)
                cpu_q = sum([i['cpus'] for i in live])
                gpu_q = sum([i['gpus'] for i in live])
                mem_q = sum([i['mem'] for i in live]) # megabytes

            nline = [f"{self._usercodes[user]}"]
            nline += list([f"{user:>{max_user_length}s}",
                           f"{stats['branch']:>{max_branch_length}s}",
                           f"{cpu_r:>8d}",
                           f"{cpu_q:>8d}",
                           f"{gpu_r:>8d}",
                           f"{gpu_q:>8d}",
                           f"{mem_r/1024:>10.0f}",
                           f"{mem_q/1024:>10.0f}",
                           f"{(str(stuck) if stuck else '-'):>7s}",
                     ])
            nline[-1] += "\033[0m"
            nline = np.array(nline, dtype='U36')
            if self.gpus_only and ((gpu_q > 0) or (gpu_r > 0)):
                self._blit(row, column, nline)
                row += 1
            elif not self.gpus_only:
                self._blit(row, column, nline)
                row += 1
        self.row = row

    def gpu_data_to_screen(self, cluster, row, column):
        if cluster not in self.cluster_stat.resource_gpu.keys():
            return
        for node in self.cluster_stat.resource_gpu[cluster]:
            for gpu_type in self.cluster_stat.resource_gpu[cluster][node]:
                res = self.cluster_stat.resource_gpu[cluster][node][gpu_type]
                # copy for the same reason as cpu_data_to_screen
                desc = self.cluster_stat.resource_gpu_desc[cluster][node][gpu_type].copy()
                not_used = np.where(res == 0)
                used = np.where(res > 0)
                desc[not_used] = self.FREE_CELL[self._node_glyph(cluster, node)]
                for i, idx in np.ndenumerate(used):
                    e = desc[idx]
                    user=e[1:].strip()
                    s_or_p = "=" if e[0] == "P" else "+"
                    desc[idx] = f"{self._usercodes[user]}{s_or_p}\033[0m"

                line =  list(f"{node:>10s}") + [" ", "("] + list(gpu_type) + [")", ":", "["] + desc.tolist() + ["\033[0m]"]
                line = np.array(line, dtype='U36')
                self._blit(row, column, line)

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
                self._blit(row, column, t) 
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
        unavailable = 0
        for node in self.cluster_stat.resource_gpu[cluster]:
            for gpu_name in self.cluster_stat.resource_gpu[cluster][node]:
                gpu_arr = self.cluster_stat.resource_gpu[cluster][node][gpu_name]
                empty_idx = np.where(gpu_arr == 0)
                n_empty = empty_idx[0].shape[0]
                # idle GPUs on a down/draining/memory-starved node are not
                # actually obtainable, so keep them out of the "available" tally
                if n_empty and (self._node_glyph(cluster, node) != "ok"):
                    unavailable += n_empty
                    continue
                if n_empty != 0:
                    if (free_nodes == 0):
                        line = list("Unallocated GPUs:")
                        self._blit(row, column, line)
                        row += 1
                    free_nodes += 1
                    range_string = ", ".join([str(i) for i in empty_idx[0]])
                    truncated_string = replace_with_ranges(range_string)
                    line = list(f"{node}:{gpu_name}:{n_empty}(IDX:{truncated_string})")
                    self.max_line_length = max(len(line), self.max_line_length)
                    self._blit(row, column, line)
                    row += 1
                    max_gpu = n_empty if n_empty > max_gpu else max_gpu
                    min_gpu = n_empty if n_empty < min_gpu else min_gpu
                    total_free += n_empty
        if free_nodes != 0:
            line = list(f"Total GPUs available: {total_free} across {free_nodes} nodes.")
            self.max_line_length = max(len(line), self.max_line_length)
            self._blit(row, column, line)
            row += 1
            line = list(f"Per node: (MIN {min_gpu}, MAX {max_gpu})")
            self.max_line_length = max(len(line), self.max_line_length)
            self._blit(row, column, line)
            row += 1
        if unavailable:
            line = list(f"({unavailable} idle GPUs unreachable: node down/draining/no memory)")
            self.max_line_length = max(len(line), self.max_line_length)
            self._blit(row, column, line)
            row += 1
        self.row = row

    def print_gpu_usage(self):
        DIV=1000 #953.674
        # below has a hard time when multi node configuration of job.
        # google "slurm how to log into a job" for multi-node suggestions..
        # https://stackoverflow.com/questions/63366098/rejoin-a-bash-slurm-job
        # Try the 'ssh=True' option in the comments section of the job submission.
        # https://portal.science.gc.ca/xwiki/bin/view/Projects/Science/Tutorials%20and%20HowTos/Quick%20Start%20to%20Using%20Linux%20Clusters%20With%20SLURM/
        out = job_smi(self.jobid, cluster="gpsc7")

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
                self._blit(row, column, t) 
            row+=1
        self.row = row + 2

    def legend_to_screen(self, row, column):
        """One-line key for the cell glyphs, drawn at the foot of each cluster."""
        cells = [
            (self.FREE_CELL['ok'], 'free'),
            ('+', 'serial'),
            ('=', 'parallel'),
            (self.FREE_CELL['mem'], 'no mem'),
            (self.FREE_CELL['drain'], 'draining'),
            (self.FREE_CELL['reserved'], 'reserved'),
            (self.FREE_CELL['down'], 'down'),
        ]
        line = []
        for glyph, label in cells:
            line.append(glyph)
            line.extend(list(f" {label}   "))
        self._blit(row, column, line)
        self.row = row + 1

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

            # this branch skips summary_to_screen, so surface downtime here too
            row = self.row + 1
            for text in self._downtime_lines(cluster):
                self._blit(row, self.max_line_length+4, list(text))
                row += 1
            self.row = row


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

        # Nothing was drawn -- e.g. --gpus-only on a cluster with no GPUs.
        # Return empty rather than emitting a legend with nothing to explain.
        if self._max_row == 0:
            return ""

        # anchor on the bottom of everything drawn, not self.row -- that is
        # whichever panel happened to finish last (the GPU list, in the right
        # column) and sits well above the foot of the CPU list on the left
        self.legend_to_screen(self._max_row + 1, 0)

        # join only the region that was actually written -- the rest is padding
        # that rstrip() would discard anyway
        live = self._screen[:self._max_row, :self._max_col]
        lines = ("".join(line).rstrip() for line in live)
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
