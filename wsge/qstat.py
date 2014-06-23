#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
qstat.py -- Helpers to serialise qstat output.

Reads SGE output as XML and parses into python dictionaries.

"""

from collections import defaultdict
from subprocess import Popen, PIPE

from lxml import etree


def qstat():
    """Return qstat output as a dictionary."""

    # full output of everything in xml
    qstat = Popen(['qstat', '-u', '*', '-f', '-xml'], stdout=PIPE).stdout.read()
    qtree = etree.fromstring(qstat)

    all_queues = {}
    users = defaultdict(defaultdict)

    #queue info
    for queue in qtree[0]:
        # queue info
        this_node = {'job_list': []}
        for qitem in queue:
            if qitem.tag == 'job_list':
                job_dict = {}
                for jitem in qitem:
                    job_dict[jitem.tag] = jitem.text
                job_dict['slots'] = int(job_dict['slots'])
                users[job_dict['JB_owner']][job_dict['JB_job_number']] = job_dict
                this_node['job_list'].append(job_dict)
            else:
                this_node[qitem.tag] = qitem.text

        #remove "all.q@' and '.local'
        this_node['name'] = this_node['name'].split('@')[1].split('.')[0]
        this_node['slots_total'] = int(this_node['slots_total'])
        all_queues[this_node['name']] = this_node

    # rest of the job info
    for wait_job in qtree[1]:
        job_dict = {}
        for jitem in wait_job:
            job_dict[jitem.tag] = jitem.text
        job_dict['slots'] = int(job_dict['slots'])
        users[job_dict['JB_owner']][job_dict['JB_job_number']] = job_dict

    return all_queues, users

