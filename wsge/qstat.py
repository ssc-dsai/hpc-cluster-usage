#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pretty summary of what's going on in the queue.

"""

import getpass
from collections import defaultdict
from pprint import pprint
from subprocess import Popen, PIPE

import numpy as np
from lxml import etree

# full output of everything in xml
qstat = Popen(['qstat', '-u', '*', '-f', '-xml'], stdout=PIPE).stdout.read()
qtree = etree.fromstring(qstat)

all_queues = {}
users = defaultdict(defaultdict)

#queue info
for queue in qtree[0]:
    # queue info
    this_node = {'job_list': []}
    jobs = []
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


# Setup how people's jobs look:
usercodes = {}
codes = range(1, 256)
for user_id, user in enumerate(users, start=1):
    usercodes[user] = "\033[38;5;%im" % user_id

# Make current user always look the same
usercodes[getpass.getuser()] = "\033[104;39;1m"

##
# Output
##

# Make a screen to manipulate since things are in columns
width = 120
height = 42
screen = np.array([[" " for _i in range(width)] for _j in range(height)],
                  dtype='S36')

# Column variables
max_head = 7
lcolumn = 0
row = 0
max_line_length = 0

##
# Cluster usage info
##


def numeric_sort(name):
    """
    Split the name into a tuple with the last two items as integers.
    """
    name = name.split('-')
    return int(name[-2]), int(name[-1])


for node in sorted(all_queues, key=numeric_sort):

    # move to next column at certain nodes
    if node in ['compute-4-0', 'compute-7-0']:
        row = 0
        lcolumn += max_line_length + 2
        max_line_length = 0

    # prune out 'compute-' to make it more compact
    nline = list("%*s " % (max_head, node.replace('compute-', ''))) + ['[']
    nusers = []
    njobs = all_queues[node]
    for job_info in njobs['job_list']:
        if job_info['slots'] > 1:
            for _slot in range(job_info['slots']):
                nusers.append((job_info['JB_owner'], 'P'))
        else:
            nusers.append((job_info['JB_owner'], 'S'))

    for user, pcode in nusers:
        if pcode == 'P':
            nline.append("%s=\033[0m" % usercodes[user])
        else:
            nline.append("%s+\033[0m" % usercodes[user])

    for empty in range(all_queues[node]['slots_total'] - len(nusers)):
        if 'state' in all_queues[node] and ('d' in all_queues[node]['state'] or
                                            'E' in all_queues[node]['state'] or
                                            'u' in all_queues[node]['state']):
            nline.append('\033[0mx\033[0m')
        else:
            nline.append('\033[0m\033[42m-\033[0m')

    # finish node
    nline.append('\033[0m]')

    nline = np.array(nline, dtype='S36')
    screen[row, lcolumn:lcolumn+len(nline)] = nline

    # increment all counters
    max_line_length = max(len(nline), max_line_length)
    row += 1

##
# Brief user info
##

row = 0
lcolumn += max_line_length + 4
max_user_length = max(len(x) for x in users)

header = list("%-*s  %-5s  %-5s  %-5s" %
              (max_user_length, 'USER', 'r', 'q', 'h'))
nline = np.array(header, dtype='S36')
screen[row, lcolumn:lcolumn+len(nline)] = nline
row += 1

for user in sorted(users):
    states = {
        'h': 0,
        'q': 0,
        'r': 0
    }
    for job, job_info in users[user].items():
        if 'h' in job_info['state']:
            states['h'] += job_info['slots']
        elif 'q' in job_info['state']:
            states['q'] += job_info['slots']
        elif 'r' in job_info['state']:
            states['r'] += job_info['slots']
        elif user == getpass.getuser():
            print "Bad job: %s" % job
        else:
            # bad job for someone else
            pass

    nline = ["%s" % usercodes[user]]
    nline += list("%-*s  %-5i  %-5i  %-5i" % (max_user_length, user,
                                              states['r'], states['q'],
                                              states['h']))
    nline += ["\033[0m"]
    nline = np.array(nline, dtype='S36')
    screen[row, lcolumn:lcolumn+len(nline)] = nline
    row += 1

print("\n".join(("".join(line)).rstrip() for line in screen))
