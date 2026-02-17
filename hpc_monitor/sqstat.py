#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
qstat.py -- Helpers to serialise qstat output.

Reads Slurm output using format strings and parses into python dictionaries.

"""

from collections import OrderedDict
from subprocess import Popen, PIPE
import json
import os
import re
import string
import pandas as pd


def expand_nodelist(nodelist_str):
    """Expand a Slurm compressed nodelist string into individual node names.

    E.g. "ib12be-[001-003,005]" -> ["ib12be-001", "ib12be-002", "ib12be-003", "ib12be-005"]
         "ib12gpu-001" -> ["ib12gpu-001"]
         "ib12be-[012,016],ib12gpu-001" -> ["ib12be-012", "ib12be-016", "ib12gpu-001"]
    """
    if not nodelist_str or nodelist_str.strip() == '':
        return []

    try:
        result = Popen(['scontrol', 'show', 'hostnames', nodelist_str],
                       stdout=PIPE, stderr=PIPE)
        out = result.stdout.read().decode('utf-8').strip()
        if out:
            return out.split('\n')
    except (FileNotFoundError, OSError):
        pass

    # Fallback: manual expansion
    return _expand_nodelist_manual(nodelist_str)


def _expand_nodelist_manual(nodelist_str):
    """Manual fallback for expanding compressed nodelists."""
    nodes = []
    # Match patterns like prefix[range] or just plain names
    pattern = re.compile(r'([^\[\],]+)\[([^\]]+)\]|([^\[\],]+)')

    for match in pattern.finditer(nodelist_str):
        if match.group(3):
            # Plain node name
            nodes.append(match.group(3).strip())
        else:
            prefix = match.group(1)
            range_str = match.group(2)
            for part in range_str.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-', 1)
                    width = len(start)
                    for i in range(int(start), int(end) + 1):
                        nodes.append(f"{prefix}{i:0{width}d}")
                else:
                    nodes.append(f"{prefix}{part}")
    return nodes


def job_smi(jobid, cluster=None):
    cluster_arg = ''
    if cluster is not None:
        cluster_arg = f'--cluster={cluster}'
    args = ['srun',
            cluster_arg,
            f'--jobid={jobid}',
            '--overlap',
            'bash -c \'echo \"NODENAME=${SLURMD_NODENAME}:${SLURM_PROCID}\" > nvidia-out.$(printf %02d $SLURM_PROCID) && nvidia-smi -q -x >> nvidia-out.$(printf %02d $SLURM_PROCID)\'',
    ]
    Popen(" ".join(args), shell=True, stdout=PIPE).stdout.read()
    # Now that we pipe output to files, need to read them.
    out = ""
    for filename in [i for i in os.listdir(".") if i.startswith("nvidia-out.")]:
        with open(filename, 'r') as f:
            out += f.read()
        f.close()
        os.remove(filename)
    return out


def sinfof(clusters):
    """Return sinfo output as a dict keyed by cluster, each containing a list of node dicts.

    Uses format-string output instead of --json for dramatically reduced output size.
    Format: %N|%X|%Y|%G|%T|%m|%e|%C
    """
    if not isinstance(clusters, str):
        c = ",".join(clusters)
    else:
        c = clusters
    sinfo_args = ['sinfo', '-M', c, '--Node', '-h',
                  '-o', '%N|%X|%Y|%G|%T|%m|%e|%C']
    sinfo_str = Popen(sinfo_args, stdout=PIPE).stdout.read()
    return _parse_sinfo_output(sinfo_str.decode('utf-8'), c)


def _parse_sinfo_output(text, clusters_arg=None):
    """Parse pipe-delimited sinfo output into a dict of cluster -> {'sinfo': [...]}.

    Handles multi-cluster output where CLUSTER: header lines separate sections.
    Returns data in a structure compatible with process_info().
    """
    result = {}
    # When querying a single cluster, Slurm may omit the CLUSTER: header.
    # Use the clusters_arg as fallback so keys match squeue output.
    current_cluster = None
    fallback_cluster = None
    if clusters_arg and ',' not in clusters_arg:
        fallback_cluster = clusters_arg

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('CLUSTER:'):
            current_cluster = line[8:].strip()
            if current_cluster not in result:
                result[current_cluster] = {'sinfo': []}
            continue

        parts = line.split('|')
        if len(parts) < 8:
            continue

        node_name = parts[0].strip()
        sockets = int(parts[1].strip())
        cores_per_socket = int(parts[2].strip())
        gres_string = parts[3].strip()
        state_str = parts[4].strip()
        memory_total = int(parts[5].strip())
        memory_free = int(parts[6].strip())
        # %C returns "allocated/idle/other/total"
        cpus_parts = parts[7].strip().split('/')
        cpus_allocated = int(cpus_parts[0])
        cpus_idle = int(cpus_parts[1])

        memory_allocated = memory_total - memory_free

        # Parse state into a list (e.g. "idle" -> ["IDLE"], "down*+drain" -> ["DOWN", "DRAIN", "NOT_RESPONDING"])
        node_states = _parse_node_state(state_str)

        node_dict = {
            'node_name': node_name,
            'sockets': sockets,
            'cores_per_socket': cores_per_socket,
            'gres_total': gres_string if gres_string != '(null)' else '',
            'node_states': node_states,
            'memory_total': memory_total,
            'memory_allocated': memory_allocated,
            'cpus_allocated': cpus_allocated,
            'cpus_idle': cpus_idle,
        }

        if current_cluster is None:
            # Single-cluster mode; use the cluster name from the -M argument
            current_cluster = fallback_cluster or '_default'
            result[current_cluster] = {'sinfo': []}

        result[current_cluster]['sinfo'].append(node_dict)

    return result


def _parse_node_state(state_str):
    """Convert sinfo long state string to a list of state tokens matching JSON format.

    Examples:
        "idle" -> ["IDLE"]
        "mixed" -> ["MIXED"]
        "down*" -> ["DOWN", "NOT_RESPONDING"]
        "idle+drain" -> ["IDLE", "DRAIN"]
        "down*+drain" -> ["DOWN", "DRAIN", "NOT_RESPONDING"]
    """
    states = []
    # The * suffix means NOT_RESPONDING
    not_responding = '*' in state_str
    state_str = state_str.replace('*', '').replace('~', '').replace('#', '').replace('$', '').replace('@', '').replace('^', '')

    for part in state_str.split('+'):
        states.append(part.strip().upper())

    if not_responding:
        states.append('NOT_RESPONDING')

    return states


def squeuef(clusters):
    """Return squeue output as a dict keyed by cluster, each containing a list of job dicts.

    Uses format-string output instead of --json for dramatically reduced output size.
    Format: %u|%g|%T|%M|%N|%C|%D|%b|%i|%tres-alloc|%tres-per-node
    """
    if not isinstance(clusters, str):
        c = ",".join(clusters)
    else:
        c = clusters
    squeue_args = ['squeue', '-M', c, '-h',
                   '-o', '%u|%g|%T|%M|%N|%C|%D|%b|%i|%tres-alloc|%tres-per-node']
    squeue_str = Popen(squeue_args, stdout=PIPE).stdout.read()
    return _parse_squeue_output(squeue_str.decode('utf-8'))


def _parse_squeue_output(text):
    """Parse pipe-delimited squeue output into a dict of cluster -> {'jobs': [...]}.

    Returns data in a structure compatible with process_jobs().
    """
    result = {}
    current_cluster = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('CLUSTER:'):
            current_cluster = line[8:].strip()
            if current_cluster not in result:
                result[current_cluster] = {'jobs': []}
            continue

        parts = line.split('|')
        if len(parts) < 11:
            continue

        user_name = parts[0].strip()
        group_name = parts[1].strip()
        job_state = parts[2].strip()
        cluster = parts[3].strip()
        node_list = parts[4].strip()
        num_cpus = int(parts[5].strip()) if parts[5].strip().isdigit() else 0
        num_nodes = int(parts[6].strip()) if parts[6].strip().isdigit() else 0
        gres = parts[7].strip()
        job_id = parts[8].strip()
        tres_alloc = parts[9].strip()
        tres_per_node = parts[10].strip()

        # Use cluster from %M field if available, else from CLUSTER: header
        if cluster and cluster != '(null)':
            effective_cluster = cluster
        else:
            effective_cluster = current_cluster or '_default'

        if effective_cluster not in result:
            result[effective_cluster] = {'jobs': []}

        job_dict = {
            'user_name': user_name,
            'group_name': group_name,
            'job_state': job_state,
            'cluster': effective_cluster,
            'node_list': node_list if node_list != '(null)' else '',
            'num_cpus': num_cpus,
            'num_nodes': num_nodes,
            'gres': gres if gres != '(null)' else '',
            'job_id': job_id,
            'tres_alloc_str': tres_alloc if tres_alloc != '(null)' else '',
            'tres_per_node': tres_per_node if tres_per_node != '(null)' else '',
        }

        result[effective_cluster]['jobs'].append(job_dict)

    return result


def sacctf(clusters, start_time, end_time, partition=""):
    """Return sacct output as a dictionary"""
    parse_format = OrderedDict({
                    "CPUTime": 30,
                    "NCPUS": 10,
                    "NNodes": 8,
                    "AllocTRES": 90,
                    "ReqTRES": 90,
                    "Elapsed": 14,
                    "JobName": 100,
                    "Account": 30,
                    "AllocNodes": 30,
                    "User": 30,
                    "Group": 30,
                    "NodeList": 30,
                    "Start": 30,
                    "End": 30,
                    "State": 10,
                    "Partition":10,
    })
    if not isinstance(clusters, str):
        c = ",".join(clusters)
    else:
        c = clusters
    format_string = ",".join([f"{k}%{v}" for k, v in parse_format.items()])
    sacct_args = ['sacct', '--allusers', '-M', c, '-S', start_time, '-E', end_time, '-o', format_string] # '--json']
    if partition:
        sacct_args += ["--partition", partition]
    sacct_str = Popen(sacct_args, stdout=PIPE).stdout.read()
    sacct_obj = _parse_sacct_pipe(sacct_str.decode('utf-8'), parse_format)
    return pd.DataFrame(sacct_obj)

def _parse_sacct_pipe(string, parse_format):
    data_lines = []
    for line in string.splitlines()[2:]:
        lcopy = line
        line_dict = {}
        for p,v in parse_format.items():
            line_dict.update({p: lcopy[:v]})
            lcopy = lcopy[v+1:]
        data_lines.append(line_dict)
    return data_lines


def sinfof_local(clusters):
    if not isinstance(clusters, str):
        c = ",".join(clusters)
    else:
        c = clusters
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "sinfo_out.txt")
    with open(filename, 'r') as f:
        txt = f.read().strip()
    return _parse_sinfo_output(txt, c)

def squeuef_local(clusters):
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "squeue_out.txt")
    with open(filename, 'r') as f:
        txt = f.read().strip()
    return _parse_squeue_output(txt)

def sacctf_local(clusters):
    parse_format = OrderedDict({
                    "CPUTime": 30,
                    "NCPUS": 10,
                    "NNodes": 8,
                    "AllocTRES": 90,
                    "ReqTRES": 90,
                    "Elapsed": 14,
                    "JobName": 100,
                    "Account": 30,
                    "AllocNodes": 30,
                    "User": 30,
                    "Group": 30,
                    "NodeList": 30,
                    "Start": 30,
                    "End": 30,
                    "State": 10,
                    "Partition":10,
    })
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "sacct_out.csv")
    with open(filename, 'r') as f:
        sacct_string = f.read()
    return pd.DataFrame(_parse_sacct_pipe(sacct_string, parse_format))
