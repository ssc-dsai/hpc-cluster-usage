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


try:
    from lxml.etree import fromstring
except ImportError:
    from xml.etree.ElementTree import fromstring


def qstat():
    """Return qstat output (job list) as a dictionary."""

    # output of jobs in xml
    qstat_str = Popen(['qstat', '-u', '*', '-xml'], stdout=PIPE).stdout.read()
    qtree = fromstring(qstat_str)

    jobs = []

    # queue info
    for job_lists in qtree:  # queue_info and job_info
        for job in job_lists:
            job_dict = {}
            for jitem in job:
                job_dict[jitem.tag] = jitem.text
            job_dict['slots'] = int(job_dict['slots'])
            jobs.append(job_dict)

    return jobs

def sinfof():
    #   sinfo -M all -a --Node --json
    sinfo_args = ['sinfo', '-M', 'all', '--Node', '--json']
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

def squeuef():
    """Return squeue output as a dictionary"""

    squeue_args = ['squeue', '-M', 'all', '--json']
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

def sinfof_local():
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "sinfo_out.json")
    with open(filename, 'r') as f:
        lines = f.readlines()
    added_braces = []
    for id, line in enumerate(lines):
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

def squeuef_local():
    filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "squeue_out.json")
    with open(filename, 'r') as f:
        lines = f.readlines()
    added_braces = []
    for id, line in enumerate(lines):
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
    return squeue_obj

def qstatf():
    """Return qstat -f output as a dictionary."""

    # full output of everything in xml
    qstat_args = ['qstat', '-u', '*', '-F', '-xml']
    qstat_str = Popen(qstat_args, stdout=PIPE).stdout.read()
    qtree = fromstring(qstat_str)

    all_queues = OrderedDict()
    users = OrderedDict()
    jobs = []

    # queue info
    for queue in qtree[0]:
        # queue info
        this_node = {'job_list': [],
                     'resource': {}}
        for qitem in queue:
            if qitem.tag == 'job_list':
                job_dict = {}
                for jitem in qitem:
                    job_dict[jitem.tag] = jitem.text
                job_dict['slots'] = int(job_dict['slots'])
                job_dict['JB_queue'] = this_node['name']
                job_dict['node'] = this_node['name'].split('@')[1].split('.')[0]
                if job_dict['JB_owner'] not in users:
                    users[job_dict['JB_owner']] = []
                users[job_dict['JB_owner']].append(job_dict)
                this_node['job_list'].append(job_dict)
                jobs.append(job_dict)
            elif qitem.tag == 'resource':
                this_node['resource'][qitem.attrib['name']] = qitem.text
            else:
                this_node[qitem.tag] = qitem.text

        # remove "all.q@' and '.local'
        node = this_node['name'].split('@')[1].split('.')[0]
        this_node['name'] = node
        this_node['slots_total'] = int(this_node['slots_total'])
        if node in all_queues:
            all_queues[node]['job_list'].extend(this_node['job_list'])
        else:
            all_queues[node] = this_node

    # rest of the job info
    for wait_job in qtree[1]:
        job_dict = {}
        for jitem in wait_job:
            job_dict[jitem.tag] = jitem.text
        job_dict['slots'] = int(job_dict['slots'])
        if job_dict['JB_owner'] not in users:
            users[job_dict['JB_owner']] = []
        users[job_dict['JB_owner']].append(job_dict)
        jobs.append(job_dict)

    return all_queues, users, jobs
