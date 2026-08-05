``hpc_monitor``
===============

Friendly terminal views and usage reports for Slurm clusters. Each CPU core and
each GPU on the cluster is drawn as a single character, coloured by the user
occupying it, so a whole federation fits on one screen.

Commands
--------

``cluster_stat``
  A live picture of the cluster: every node's cores and GPUs, who is on them,
  what is free, and what cannot be used. Add ``-g``/``--gpus-only`` to show just
  the GPU nodes, or ``-M``/``--clusters`` to pick a cluster (default ``all``).

``user_report``
  Usage history from the accounting database, per user, group or agency. Prints
  a summary table, draws a stacked usage-over-time chart in the terminal for
  focused reports, and writes a PNG with ``--plot``/``-o``.

``gpu_report``
  Histogram of GPUs-per-job over a date range, stacked by agency.

``gpu_usage <jobid>``
  Live ``nvidia-smi`` core and memory utilisation for a running job, as progress
  bars per GPU.

``cluster_stat``, ``user_report`` and ``gpu_report`` take ``-M``/``--clusters``
and ``-P``/``--partition``; the two reporting commands also take
``-S``/``--start-time``, ``-E``/``--end-time`` and ``-o``/``--output``.
``--help`` lists everything.

Dates
~~~~~

``-S`` and ``-E`` accept several forms:

======================  ==========================================================
``100725``              the original ``mmddyy``
``2025-10-07``          a calendar date; ``Oct 7 2025`` and ``1 August 2026`` work
``now``, ``today``      this instant, or midnight at the start of today
``yesterday``           and ``tomorrow``
``last week``           seven days back
``3 days ago``          also ``2 weeks``, ``6 months ago``, ``two weeks ago``
``2w``                  shorthand: ``d``, ``w``, ``m``, ``y``, and ``h`` for hours
``this month``          start of the current month; also ``this week``/``this year``
``last friday``         the most recent past occurrence of that weekday
======================  ==========================================================

Relative expressions are rolling offsets from now -- ``last week`` is seven days
back rather than the previous calendar week -- so the two ends of a range
compose predictably::

    user_report -M gpsc7 -S "last week"
    user_report -M gpsc7 -S "3 months ago" -E "1 month ago" --by agency

Everything except ``now`` and ``N hours ago`` lands on midnight at the start of
the named day. **The end of the window is exclusive**, so ``-E today`` stops at
midnight this morning and leaves today out. Use ``-E now`` -- which is the
default, so simply omitting ``-E`` is usually what you want.

The commands share one argument parser, so ``--help`` shows every option
regardless of which command you ask. Options outside a command's own group do
not apply to it.

Installation
------------

Setup as a Python package so that the modules are in your ``PYTHONPATH`` and the
console scripts are in your ``PATH``. In the source directory:

.. code-block::

    pip install --user .

That covers ``cluster_stat`` and ``gpu_usage``, which need only ``numpy``,
``pyfiglet``, ``xmltodict`` and ``python-dateutil``.

``gpu_report`` and ``user_report`` additionally draw plots, which needs pandas,
matplotlib and seaborn. They are an optional extra rather than a hard
requirement, because they cost roughly 1.8s of interpreter startup and are
imported lazily so that ``cluster_stat`` never pays for them:

.. code-block::

    pip install --user ".[reports]"

Packaging is a single ``pyproject.toml`` (PEP 517/518 build system, PEP 621
metadata); there is no ``setup.py``. Build a distribution with ``python -m
build``.

``cluster_stat``
----------------

Colours distinguish users; your own jobs are highlighted. Excerpt with the
colour codes stripped::

   ib13be-001:[                                        ]           used:  954
   ib13be-002:[                                        ]      available: 3086
   ib13be-003:[                                        ]          total: 4120
   ib13be-004:[                                        ]    unreachable:   80  (2 mem)
   ib13be-005:[============================= =====   ==]         memory:  3.0/19.9 TB (15%)
   ib13be-006:[==============================          ]     freeing 1h:    0 CPU, 0 GPU (0 jobs)
   ib13be-007:[========================================]    downtime: Sat 08 Aug 12:30 +1d12h (whole cluster)
   ib13be-008:[================    ==========          ]       max walltime until then: 2d20h
   ib13be-009:[==========                              ]
   ib13be-010:[                                        ]    queued: 2 runnable, 0 blocked

     free   + serial   = parallel   m no mem   d draining   r reserved   X down

Each character inside the brackets is one core:

===========  =================================================================
``(space)``  free and actually allocatable
``+``        allocated to a single-core, single-node job
``=``        allocated to a parallel job
``m``        idle, but the node has no memory left, so nothing can land here
``d``        node is draining: existing jobs continue, no new ones start
``r``        node is reserved
``X``        node is down or not responding
===========  =================================================================

The distinction matters: a node can show idle cores while being unusable. Only
the blank cells count towards ``available``; everything else is reported
separately as ``unreachable``.

The right-hand panel summarises the cluster and lists per-user totals::

         USER AGENCY   CPU_R   CPU_Q   GPU_R   GPU_Q  MEM_R/GB  MEM_Q/GB  STUCK
       fuf000   AAFC       0       1       0       0         0         2      -
       jfb001  NRCAN      10      10       0       0        35        35      -
       maz002  NRCAN     370       0       0       0      1280         0      -

``_R`` is running, ``_Q`` is queued. ``STUCK`` counts pending jobs that cannot
run without someone intervening (unsatisfiable dependencies, held jobs, invalid
QOS); these are excluded from the ``_Q`` columns so the queue figures reflect
real contention rather than abandoned work.

``downtime`` is the next ``MAINT`` reservation. Because Slurm will not start a
job that would still be running when the window opens, ``max walltime until
then`` is the longest ``--time`` worth requesting right now.

GPU nodes are drawn the same way, one character per GPU::

    ib14gpu-001 (a100-sxm4-40gb):[+===+ = ]
    ib14gpu-002 (a100-sxm4-40gb):[===== = ]
    ib14gpu-005 (a100-sxm4-40gb):[= ===   ]

    Unallocated GPUs:
    ib14gpu-001:a100-sxm4-40gb:2(IDX:5,7)

``user_report``
---------------

.. code-block::

    user_report -M gpsc7 -S 072026 -E 080526 -n 6 --plot -o usage.png

::

    Usage by user -- gpsc7 -- 20 Jul 2026 to 05 Aug 2026
    130 users, 403,756 job allocations, bucketed by day

    USER    AGENCY     JOBS     CPU_HRS   GPU_HRS  AVG_WAIT  MAX_JOB  FAIL%
    -----------------------------------------------------------------------
    mob001  NRC          23     414,377     1,669        4m  512c/8g  39.1%
    yow001  DFO          53     223,528         0        0m     256c  39.2%
    sdfo004 DFO         575     207,942         0       13m     384c  18.1%
    -----------------------------------------------------------------------
    +124 others      402233     759,597     4,500
    -----------------------------------------------------------------------
    TOTAL            403756   1,987,251     6,168

``-n``/``--top`` limits the table (``0`` for all). With ``--plot`` or ``-o`` a
PNG is written showing consumption stacked over time; CPU-hours and GPU-hours
get their own axes.

Time buckets adapt to the window: daily up to five weeks, weekly up to eight
months, monthly beyond that.

Focused reports
~~~~~~~~~~~~~~~

Narrow the report and it draws a stacked chart in the terminal as well:

==========================  ====================================================
``-u``/``--users``          restrict to a comma separated list of users
``-G``/``--group``          restrict to groups (``dfo_chs_enav``) or whole
                            agencies (``DFO``), case-insensitively
``-b``/``--by``             what a row represents: ``user`` (default), ``group``
                            or ``agency``
``-m``/``--measure``        what the terminal chart plots: ``cpu`` (default),
                            ``gpu``, or ``jobs`` started
``-c``/``--chart``          draw the chart even for an unfiltered report
``--no-chart``              suppress it
==========================  ====================================================

Every user in a group, with their stats::

    user_report -M gpsc7 -G dfo_chs_enav

Which departments inside an agency are consuming the time::

    user_report -M gpsc7 -G DFO --by group

::

     54,034 ┤++++++        ###### ###### ###### ###### ++++++ ++++++ ++++++ ~~~~~~
            │====== ~~~~~~ ###### ###### ###### ###### ====== ====== ====== ~~~~~~
     27,017 ┤###### ====== ###### ###### ###### ###### ###### ====== ====== ======
            │###### ###### ###### ###### ###### ###### ###### ###### ###### ######
          0 └──────────────────────────────────────────────────────────────────────
             26 Jul 27 Jul 28 Jul 29 Jul 30 Jul 31 Jul 01 Aug 02 Aug 03 Aug 04 Aug

      ## dfo_bioios   == dfo_comda   ++ dfo_dpnm   ~~ dfo_chs_enav   :: dfo_orph

    GROUP        AGENCY   USERS    JOBS     CPU_HRS   GPU_HRS  AVG_WAIT  MAX_JOB  FAIL%
    -----------------------------------------------------------------------------------
    dfo_bioios   DFO          6    3321     436,757         0        4m     384c  62.8%
    dfo_comda    DFO         17    6929     154,965         0       12m     512c   3.6%
    dfo_dpnm     DFO         10    1822      73,000         0     1h05m     256c   2.4%
    dfo_chs_enav DFO          2  136956      40,805         0        0m       7c   0.0%

Groups are ``<agency>_<department>``, so ``--by agency`` rolls ``dfo_bioios``
and ``dfo_comda`` together under ``DFO`` -- the same split ``cluster_stat`` uses
for its AGENCY column.

In a colour terminal each series is a colour; when output is piped or
redirected, each series gets its own fill character instead, so the stack stays
readable either way. Bars are apportioned by largest remainder, so the segments
always sum to the drawn bar height, and the top of each bar uses an eighth-block
so totals resolve to better than one row. Column count and tick spacing adapt to
the terminal width.

How usage is counted
~~~~~~~~~~~~~~~~~~~~

Reports like this are easy to get wrong, so the accounting is deliberate:

* **Hours land in the period they were consumed**, not the period the job
  started in. Each job's CPU/GPU-seconds are spread across the buckets its
  runtime overlaps and clipped to the reporting window, so a two-week job
  contributes to every day it ran rather than dumping its whole cost on day one.

* **Job steps are excluded** (``sacct -X``). Without this, ``.batch``,
  ``.extern`` and numbered steps come back as separate rows — around two thirds
  of all rows — inflating every per-job statistic and making the query far
  slower.

* **Stale records are excluded.** When a job dies without ``slurmdbd`` recording
  an end, ``sacct`` keeps reporting it ``RUNNING`` with ``End=Unknown`` and an
  ``Elapsed`` that grows against the wall clock on every query. Such jobs are
  identified by cross-referencing ``squeue``, left out of the totals, and
  reported in the header. Left in, they can invent a large fraction of a
  cluster's apparent usage.

* ``FAIL%`` counts ``FAILED``, ``TIMEOUT``, ``OUT_OF_MEMORY``, ``NODE_FAIL``,
  ``BOOT_FAIL`` and ``DEADLINE`` against finished jobs. ``CANCELLED`` is not a
  failure — that is normally someone changing their mind.

A useful sanity check when extending any of this: total CPU-hours in a period
must not exceed the cluster's core count times the length of the period.

``gpu_report``
--------------

Histogram of how many GPUs each job asked for over a date range, stacked by
agency, on a log count axis::

    gpu_report -M gpsc7 -S 070126 -E 080126 -o gpus_per_job.png

Answers whether the GPU nodes are being used for genuinely multi-GPU work or
mostly for single-GPU jobs.

``gpu_usage``
-------------

Runs ``nvidia-smi`` inside a running job and renders per-GPU core and memory
utilisation as progress bars::

    gpu_usage 7506338

Takes the job ID as a positional argument. The job must be running, and this
works best for single-node jobs.

Note that the cluster is currently hardcoded to ``gpsc7``; ``-M`` has no effect
on this command.

Performance
-----------

``sacct`` is the slow part, and it scales with the number of job records rather
than the size of the cluster. On a busy cluster expect roughly 12 seconds for a
week and 30 seconds for a month, so ``user_report`` and ``gpu_report`` are
report commands rather than interactive ones. (Without ``-X`` the same one-week
query takes over three minutes.)

``cluster_stat`` queries only ``sinfo``, ``squeue`` and ``scontrol`` and returns
in about two seconds for a four-cluster federation.

Credit
------

Originally ``wsge`` by Tom Daff, for GridEngine. This version targets Slurm and
has been substantially rewritten.
