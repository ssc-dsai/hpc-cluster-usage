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
import string

# NB pandas is imported lazily inside the sacct helpers; it is not needed by the
# sinfo/squeue path that drives the cluster_stat display.

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
    if not isinstance(clusters, str):
        c = ",".join(clusters)
    else:
        c = clusters
    sinfo_args = ['sinfo', '-M', c, '--Node', '--json']
    sinfo_str = Popen(sinfo_args, stdout=PIPE).stdout.read()
    return _read_json_string(sinfo_str.decode('utf-8'))
    #added_braces = []
    #for id, line in enumerate(sinfo_str.decode('utf-8').split("\n")):
    #    nline = line.strip()
    #    if nline.startswith('CLUSTER'):
    #        cid = "\""+nline.split(":")[-1].strip()+"\""
    #        if id != 0:
    #            #added_braces.append("},".rjust(5))
    #            added_braces[-1] = added_braces[-1] + ","
    #            added_braces.append(f"{cid}:")
    #        else:
    #            added_braces.append("{")
    #            added_braces.append(f"{cid}:")

    #    elif (nline):
    #        added_braces.append(nline)

    #added_braces.append("}")
    #sinfo_obj = json.loads("\n".join(added_braces))
    #return sinfo_obj

def squeuef(clusters):
    """Return squeue output as a dictionary"""

    if not isinstance(clusters, str):
        c = ",".join(clusters)
    else:
        c = clusters
    squeue_args = ['squeue', '-M', c, '--json']
    squeue_str = Popen(squeue_args, stdout=PIPE).stdout.read()
    return _read_json_string(squeue_str.decode('utf-8'))

    # have to insert curly braces at the outer level (CLUSTER)
    #added_braces = []
    #for id, line in enumerate(squeue_str.decode('utf-8').split("\n")):
    #    nline = line.strip()
    #    if nline.startswith('CLUSTER'):
    #        cid = "\""+nline.split(":")[-1].strip()+"\""
    #        if id != 0:
    #            #added_braces.append("},".rjust(5))
    #            added_braces[-1] = added_braces[-1] + ","
    #            added_braces.append(f"{cid}:")
    #        else:
    #            added_braces.append("{")
    #            added_braces.append(f"{cid}:")

    #    elif (nline):
    #        added_braces.append(nline)

    #added_braces.append("}")
    #squeue_obj = json.loads("\n".join(added_braces))
    # node
    #   --- slots_total
    #   --- state
    # job_list
    #   --- slots
    #   --- JB_owner
    #return squeue_obj

USAGE_FIELDS = ("JobID", "User", "Account", "Group", "Partition", "State",
                "Submit", "Start", "End", "Elapsed", "AllocTRES", "NNodes")


def sacct_usagef(clusters, start_time, end_time, partition=""):
    """Yield one dict per job allocation over the window.

    Two deliberate differences from sacctf():

    -X restricts output to job allocations. Without it sacct also returns every
    step (.batch, .extern, .0) as its own row -- 67% of rows over a sample week
    -- which both triples the row count and inflates any per-job statistic.

    --parsable2 gives '|' separated fields instead of fixed-width columns, so
    there are no %width values to overflow and no truncated values to guess at.
    """
    if not isinstance(clusters, str):
        clusters = ",".join(clusters)

    args = ['sacct', '--allusers', '-X', '--parsable2', '--noheader',
            '-M', clusters, '-S', start_time, '-E', end_time,
            '-o', ",".join(USAGE_FIELDS)]
    if partition:
        args += ["--partition", partition]

    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"sacct failed: {err.decode('utf-8', 'replace').strip()}")

    rows = []
    for line in out.decode('utf-8', 'replace').splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < len(USAGE_FIELDS):
            continue
        rows.append(dict(zip(USAGE_FIELDS, parts)))
    return rows


def sacct_usagef_local(clusters, start_time, end_time, partition=""):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "usage_out.psv")
    if not os.path.exists(filename):
        return []
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= len(USAGE_FIELDS):
                rows.append(dict(zip(USAGE_FIELDS, parts)))
    return rows


def live_job_idsf(clusters):
    """Job IDs the controller currently knows about, or None if it cannot say.

    Needed to tell a genuinely running job from a stale accounting record. When
    a job dies without slurmdbd recording an end, sacct keeps reporting it as
    RUNNING with End=Unknown and an Elapsed that grows on every query -- worth
    millions of phantom CPU-hours if taken at face value.

    None (rather than an empty set) signals "could not check", so callers do
    not mistake a squeue failure for "nothing is running".
    """
    if not isinstance(clusters, str):
        clusters = ",".join(clusters)
    args = ['squeue', '-M', clusters, '-h', '-o', '%i %F']
    try:
        proc = Popen(args, stdout=PIPE, stderr=PIPE)
    except OSError:
        return None
    out, _ = proc.communicate()
    if proc.returncode != 0:
        return None

    ids = set()
    for line in out.decode('utf-8', 'replace').splitlines():
        for token in line.split():
            token = token.strip()
            if not token:
                continue
            ids.add(token)
            # 1234_5 (array task) should also match on its parent id
            ids.add(token.split("_")[0])
    return ids


def reservationf(clusters):
    """Return {cluster: [reservation, ...]} for the given clusters.

    scontrol refuses more than one cluster per invocation, so fan out one
    process per cluster and collect them together. Each call is ~60ms and they
    overlap, so this stays cheap even on a federation.
    """
    if isinstance(clusters, str):
        clusters = [c for c in clusters.split(",") if c]

    procs = {}
    for cluster in clusters:
        args = ['scontrol', '-M', cluster, 'show', 'reservation', '--json']
        try:
            procs[cluster] = Popen(args, stdout=PIPE, stderr=PIPE)
        except OSError:
            # no scontrol on PATH -- treat downtime info as simply unavailable
            return {}

    results = {}
    for cluster, proc in procs.items():
        out, _ = proc.communicate()
        if proc.returncode != 0:
            continue
        try:
            results[cluster] = json.loads(out.decode('utf-8')).get('reservations', [])
        except (json.decoder.JSONDecodeError, UnicodeDecodeError):
            # older Slurm without --json support, or a partial read
            continue
    return results


def reservationf_local(clusters):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resv_out.json")
    if not os.path.exists(filename):
        return {}
    with open(filename, 'r') as f:
        return json.load(f)


def sacctf(clusters, start_time, end_time, partition=""):
    """Return sacct output as a dictionary"""
    import pandas as pd
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
    # -X: allocations only. Without it sacct also emits every job step
    # (.batch/.extern/.N) as its own row -- ~67% of rows over a sample week --
    # which counted each step as a separate job in the histogram and made the
    # query roughly 15x slower.
    sacct_args = ['sacct', '--allusers', '-X', '-M', c, '-S', start_time, '-E', end_time, '-o', format_string] # '--json']
    if partition:
        sacct_args += ["--partition", partition]
    #print(" ".join(sacct_args))
    sacct_str = Popen(sacct_args, stdout=PIPE).stdout.read()
    #print(sacct_str.decode('utf-8'))
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

def _read_json_string(txt):
    """Optimized JSON string parser for Slurm output."""
    # Try to parse as single JSON first (faster path)
    try:
        return json.loads(txt)
    except json.decoder.JSONDecodeError:
        pass
    
    # Fallback to multi-cluster parsing with optimized approach
    return_dict = {}
    lines = txt.split('\n')
    current_cluster = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('CLUSTER:'):
            # Process previous cluster if exists
            if current_cluster and current_content:
                try:
                    json_content = '\n'.join(current_content)
                    json_dict = json.loads(json_content)
                    return_dict[current_cluster] = json_dict
                except json.decoder.JSONDecodeError:
                    pass
            
            # Start new cluster
            current_cluster = line[8:].strip()  # Remove 'CLUSTER:' prefix
            current_content = []
        elif line and current_cluster:
            current_content.append(line)
    
    # Process final cluster
    if current_cluster and current_content:
        try:
            json_content = '\n'.join(current_content)
            json_dict = json.loads(json_content)
            return_dict[current_cluster] = json_dict
        except json.decoder.JSONDecodeError:
            pass
    
    return return_dict


def sinfof_local(clusters):
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "sinfo_out.json")
    with open(filename, 'r') as f:
        txt = f.read().strip()
    return _read_json_string(txt)

def squeuef_local(clusters):
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "squeue_out.json")
    with open(filename, 'r') as f:
        txt = f.read().strip()
    return _read_json_string(txt)
    
def sacctf_local(clusters):
    import pandas as pd
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
