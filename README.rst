``wsge``
--------

Some tools to make friendly outputs of GridEngine job information. A few of
the new commands:

* ``cluster_stat``

  * An overview of jobs currently running on the cluster and who is
    running them.

* ``wstat``

  * A tweaked qstat that shows the full job name and has age of jobs instead
    of time.

* ``scrpit-submit``

  * Submit generic jobs to the cluster. Includes tweaks to keep parallel
    jobs within the same node as much as possible.


Installation
============

Setup as a Python package so that the modules are in your ``PYTHONPATH`` and
individual scripts are in your ``PATH``. In the source directory:

.. code-block::

    pip install --user .

If ``lxml`` is installed it will be used for a slight speed-up.

Outputs
=======

Note that colours are used in the terminal output to distinguish
between different users.

``cluster_stat``::

     node05 [================            ]        used:           296
     node06 [================            ]        available:       52
     node07 [============================]        total:          348
     node08 [================            ]
     node09 [================++++++++++++]        USER    r      q      h
     node15 [================]                    ab218   64     0      0
     node16 [================================]    xdg30   16     0      0
     node17 [================]                    vdn81   32     32     0
     node18 [================]                    ov415   60     0      0
     node19 [========        ]                    vsm34   12     16     0
     node20 [================]                    acv24   112    32     0
     node21 [========        ]
     node22 [================]
     node23 [================]
     node24 [================]
     node25 [================]
     node26 [================]

``wstat -f``::

    job-ID   prior      user          state  run/wait time       queue            slots  name
    
    >> node5        Slots:  [||||||||||||||||||||||||....] 24/28  Load: 6.64
    >>              Memory: [|...................] 5.539G/62.813G
    1883666  15.00484   tyd34         r                 1:50:37  short               16  search_chris
    1883694  15.00226   tr554         dr                0:01:07  short                8  cell_migration
    
    >> node6        Slots:  [||||||||||||||||||||||||....] 24/28  Load: 6.60
    >>              Memory: [|...................] 5.666G/62.813G
    1883667  15.00484   tyd34         r                 1:48:07  short               16  search_01c
    1883693  15.00226   tr554         dr                0:01:07  short                8  cell_migration

    >> node7        Slots:  [||||||||||||||||||||||||||||] 28/28  Load: 28.02
    >>              Memory: [|...................] 3.633G/62.813G
    1883643  15.00871   uy415         r                 5:24:37  short               28  test

    >> node8        Slots:  [||||||||||||||||............] 16/28  Load: 15.46
    >>              Memory: [|...................] 3.695G/62.813G
    1880531  15.00484   tyd34         r                22:29:22  short               16  MQG_125




