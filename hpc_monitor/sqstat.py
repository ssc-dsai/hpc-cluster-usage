#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
qstat.py -- Helpers to serialise qstat output.

Reads SGE output as XML and parses into python dictionaries.

"""

from collections import OrderedDict
from subprocess import Popen, PIPE
import json
import os
import re

def job_smi(jobid, cluster=None):
    cluster_arg = ''
    if cluster is not None:
        cluster_arg = f'--cluster={cluster}'
    args = ['srun',
            cluster_arg,
            f'--jobid={jobid}',
            '--overlap',
            '--gres=gpu:8',
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
    if not isinstance(clusters, str):
        c = ",".join(clusters)
    else:
        c = clusters
    sinfo_args = ['sinfo', '-M', c, '--Node', '--json']
    sinfo_str = Popen(sinfo_args, stdout=PIPE).stdout.read()
    added_braces = []
    for id, line in enumerate(sinfo_str.decode('utf-8').split("\n")):
        nline = line.strip()
        if nline.startswith('CLUSTER'):
            cid = "\""+nline.split(":")[-1].strip()+"\""
            if id != 0:
                #added_braces.append("},".rjust(5))
                added_braces[-1] = added_braces[-1] + ","
                added_braces.append(f"{cid}:")
            else:
                added_braces.append("{")
                added_braces.append(f"{cid}:")

        elif (nline):
            added_braces.append(nline)

    added_braces.append("}")
    sinfo_obj = json.loads("\n".join(added_braces))
    return sinfo_obj

def squeuef(clusters):
    """Return squeue output as a dictionary"""

    if not isinstance(clusters, str):
        c = ",".join(clusters)
    else:
        c = clusters
    squeue_args = ['squeue', '-M', c, '--json']
    squeue_str = Popen(squeue_args, stdout=PIPE).stdout.read()
    # have to insert curly braces at the outer level (CLUSTER)
    added_braces = []
    for id, line in enumerate(squeue_str.decode('utf-8').split("\n")):
        nline = line.strip()
        if nline.startswith('CLUSTER'):
            cid = "\""+nline.split(":")[-1].strip()+"\""
            if id != 0:
                #added_braces.append("},".rjust(5))
                added_braces[-1] = added_braces[-1] + ","
                added_braces.append(f"{cid}:")
            else:
                added_braces.append("{")
                added_braces.append(f"{cid}:")

        elif (nline):
            added_braces.append(nline)

    added_braces.append("}")
    squeue_obj = json.loads("\n".join(added_braces))
    # node
    #   --- slots_total
    #   --- state
    # job_list
    #   --- slots
    #   --- JB_owner
    return squeue_obj

def sinfof_local(clusters):
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "sinfo_out.json")
    with open(filename, 'r') as f:
        lines = f.readlines()
    added_braces = []
    for id, line in enumerate(lines):
        nline = line.strip()
        if nline.startswith('CLUSTER'):
            cid = "\""+nline.split(":")[-1].strip()+"\""
            if cid not in clusters:
                continue
            if id != 0:
                #added_braces.append("},".rjust(5))
                added_braces[-1] = added_braces[-1] + ","
                added_braces.append(f"{cid}:")
            else:
                added_braces.append("{")
                added_braces.append(f"{cid}:")

        elif (nline):
            added_braces.append(nline)

    added_braces.append("}")
    sinfo_obj = json.loads("\n".join(added_braces))
    return sinfo_obj

def squeuef_local(clusters):
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "squeue_out.json")
    with open(filename, 'r') as f:
        lines = f.readlines()
    added_braces = []
    for id, line in enumerate(lines):
        nline = line.strip()
        if nline.startswith('CLUSTER'):
            cid = "\""+nline.split(":")[-1].strip()+"\""
            if cid not in clusters:
                continue
            if id != 0:
                #added_braces.append("},".rjust(5))
                added_braces[-1] = added_braces[-1] + ","
                added_braces.append(f"{cid}:")
            else:
                added_braces.append("{")
                added_braces.append(f"{cid}:")

        elif (nline):
            added_braces.append(nline)

    added_braces.append("}")
    squeue_obj = json.loads("\n".join(added_braces))
    return squeue_obj

