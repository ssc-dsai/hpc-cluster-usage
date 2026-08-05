"""Per-user usage history from the Slurm accounting database.

Prints a per-user summary table and, with --plot/-o, writes a stacked
consumption-over-time chart.

Resource-hours are attributed to the period in which they were *consumed*, not
to the period a job happened to start in: each job's CPU/GPU-seconds are spread
across the buckets its runtime actually overlaps, and clipped to the reporting
window. A two-week job therefore contributes to every day it ran rather than
dumping its whole cost on day one.
"""

import os
import re
import sys
import getpass
from collections import OrderedDict, defaultdict
from math import ceil
from datetime import datetime, timedelta

from .sqstat import sacct_usagef, sacct_usagef_local, live_job_idsf

# Categorical palette, fixed slot order -- assigned by rank and never cycled.
# Validated for the light chart surface (lightness band, chroma floor, adjacent
# CVD separation, normal-vision floor). Three slots sit below 3:1 contrast on
# this surface, which is permitted here because the command always prints the
# table view alongside the chart.
SERIES_COLOURS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
# Deliberately neutral: "Other" is an aggregate, not an identity, so it must not
# read as another named user.
OTHER_COLOUR = "#8a8a85"

# Terminal equivalents, snapped to the xterm-256 cube and re-validated against a
# dark terminal surface rather than assumed: the naive nearest-RGB mapping
# collapsed two slots onto the same index and broke the lightness band. These six
# pass the band, chroma floor, adjacent CVD separation and normal-vision floor
# as a set (worst adjacent CVD dE 11.9, normal-vision 23.8). Six rather than
# eight because a stacked chart 14 rows tall cannot resolve more than that.
TERM_COLOURS = [32, 167, 31, 136, 169, 28]
TERM_OTHER = 102
TERM_RESET = "\033[0m"

# Eighths of a cell, for sub-row resolution at the top of a bar.
PARTIAL_BLOCKS = " ▁▂▃▄▅▆▇"

# Secondary encoding for when there is no colour (piped output, --no-colour
# terminals): identity must not rest on hue alone, so each slot also gets a
# distinct fill texture.
TERM_GLYPHS = ["#", "=", "+", "~", ":", "-"]
TERM_OTHER_GLYPH = "."

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"

# States that mean the work was lost. CANCELLED is excluded: that is normally a
# person changing their mind, not a failure of the job.
FAILED_STATES = frozenset({"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT",
                           "BOOT_FAIL", "DEADLINE"})
# Jobs that never ran contribute no hours and would skew the wait statistics.
UNSTARTED_STATES = frozenset({"PENDING"})


def parse_elapsed(text):
    """Seconds from sacct's [DD-[HH:]]MM:SS elapsed format."""
    text = (text or "").strip()
    if not text:
        return 0
    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        try:
            days = int(day_part)
        except ValueError:
            return 0
    bits = text.split(":")
    try:
        bits = [int(b) for b in bits]
    except ValueError:
        return 0
    if len(bits) == 3:
        hours, minutes, seconds = bits
    elif len(bits) == 2:
        hours, (minutes, seconds) = 0, bits
    elif len(bits) == 1:
        hours, minutes, seconds = 0, 0, bits[0]
    else:
        return 0
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_timestamp(text):
    """Epoch seconds from sacct's ISO timestamps; None for Unknown/None/blank."""
    text = (text or "").strip()
    if not text or text in ("Unknown", "None", "N/A"):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").timestamp()
    except ValueError:
        return None


_TRES_CPU = re.compile(r'(?:^|,)cpu=(\d+)')
_TRES_GPU = re.compile(r'gres/gpu=(\d+)')
_TRES_NODE = re.compile(r'(?:^|,)node=(\d+)')


def parse_tres(text):
    """CPU, GPU and node counts out of an AllocTRES string."""
    text = text or ""
    cpu = _TRES_CPU.search(text)
    # gres/gpu=N is the total; gres/gpu:<model>=N also appears and would double
    # count, so take the largest single match rather than summing them
    gpus = [int(m) for m in _TRES_GPU.findall(text)]
    node = _TRES_NODE.search(text)
    return (int(cpu.group(1)) if cpu else 0,
            max(gpus) if gpus else 0,
            int(node.group(1)) if node else 0)


def normalise_state(text):
    """'CANCELLED by 12345' -> 'CANCELLED'."""
    return (text or "").strip().split(" ")[0].upper()


def split_affiliation(group, account):
    """(group, agency) for a row.

    Slurm groups here are '<agency>_<department>' -- dfo_chs_enav, nrcan_gencfl,
    ssc_dsai. The leading segment is the agency, which is the same split
    cluster_stat uses for its AGENCY column. Account is the fallback when the
    group is blank.
    """
    group = (group or "").strip() or (account or "").strip()
    agency = group.split("_")[0].upper() if group else ""
    return group, agency


def bucket_size(start, end):
    """Pick a period length that yields a readable number of columns."""
    span_days = max(1, (end - start).days)
    if span_days <= 35:
        return "day"      # up to ~5 weeks of daily bars stays readable
    if span_days <= 240:
        return "week"
    return "month"


def bucket_bounds(when, size):
    """(label, start, end) of the bucket containing `when`."""
    day = datetime(when.year, when.month, when.day)
    if size == "day":
        return day.strftime("%d %b"), day, day + timedelta(days=1)
    if size == "week":
        monday = day - timedelta(days=day.weekday())
        return monday.strftime("%d %b"), monday, monday + timedelta(days=7)
    first = datetime(when.year, when.month, 1)
    nxt = datetime(when.year + (when.month == 12), (when.month % 12) + 1, 1)
    return first.strftime("%b %Y"), first, nxt


def build_buckets(start, end, size):
    """Ordered list of (label, start_epoch, end_epoch) covering the window."""
    buckets = []
    cursor = start
    while cursor < end:
        label, b_start, b_end = bucket_bounds(cursor, size)
        buckets.append((label, b_start.timestamp(), b_end.timestamp()))
        cursor = b_end
    return buckets


class UsageHistory:
    """Aggregate sacct rows into per-user totals and a per-period series."""

    def __init__(self, **kwargs):
        # Drop any argparse dest that collides with a method name: --plot would
        # otherwise bind self.plot to True and shadow write_plot's sibling.
        self.__dict__.update({k: v for k, v in kwargs.items()
                              if not callable(getattr(type(self), k, None))})
        self.rows = []
        self.users = OrderedDict()
        self.buckets = []
        self.series = defaultdict(lambda: defaultdict(float))
        self.gpu_series = defaultdict(lambda: defaultdict(float))
        self.job_series = defaultdict(lambda: defaultdict(float))
        self.skipped = 0
        self.stale = 0
        self.live_ids = None

    def fetch(self):
        if getattr(self, "local", False):
            self.rows = sacct_usagef_local(self.clusters, None, None)
            self.live_ids = None
        else:
            self.rows = sacct_usagef(self.clusters,
                                     self.start_time.strftime('%m%d%y'),
                                     self.end_time.strftime('%m%d%y'),
                                     partition=getattr(self, "partition", ""))
            self.live_ids = live_job_idsf(self.clusters)
        return self.rows

    def is_stale(self, job_id, state, end):
        """True for a RUNNING record the controller no longer knows about.

        Such a job died without an end being written to the accounting DB. Its
        Elapsed keeps growing against wall-clock, so charging it any hours would
        invent usage that never happened.
        """
        if state != "RUNNING" or end is not None:
            return False
        if self.live_ids is None:
            return False  # could not check; trust the record rather than guess
        job_id = (job_id or "").strip()
        return not (job_id in self.live_ids
                    or job_id.split("_")[0] in self.live_ids)

    def process(self):
        size = bucket_size(self.start_time, self.end_time)
        self.bucket_size = size
        self.buckets = build_buckets(self.start_time, self.end_time, size)
        window_start = self.start_time.timestamp()
        window_end = self.end_time.timestamp()
        now = datetime.now().timestamp()

        wanted = getattr(self, "users_filter", None)

        by = getattr(self, "by", "user") or "user"
        wanted_group = getattr(self, "group_filter", None)

        for row in self.rows:
            user = row['User'].strip()
            if not user:
                self.skipped += 1
                continue
            if wanted and user not in wanted:
                continue

            group, agency = split_affiliation(row['Group'], row['Account'])
            # -G accepts either a full group (dfo_chs_enav) or an agency (DFO)
            if wanted_group and not ({group.lower(), agency.lower()} & wanted_group):
                continue

            # what one row of the report represents
            key = {"user": user, "group": group, "agency": agency}.get(by, user)
            if not key:
                self.skipped += 1
                continue

            state = normalise_state(row['State'])
            cpus, gpus, nodes = parse_tres(row['AllocTRES'])
            submit = parse_timestamp(row['Submit'])
            start = parse_timestamp(row['Start'])
            end = parse_timestamp(row['End'])
            elapsed = parse_elapsed(row['Elapsed'])

            stats = self.users.setdefault(key, {
                'user': key,
                'account': row['Account'].strip(),
                'group': group,
                'agency': agency,
                'members': set(),
                'jobs': 0, 'failed': 0, 'finished': 0,
                'cpu_hours': 0.0, 'gpu_hours': 0.0,
                'max_cpus': 0, 'max_gpus': 0, 'max_nodes': 0,
                'wait_total': 0.0, 'wait_count': 0, 'stale': 0,
            })
            stats['members'].add(user)
            stats['jobs'] += 1
            if state in FAILED_STATES:
                stats['failed'] += 1
                stats['finished'] += 1
            elif state in ("COMPLETED", "CANCELLED", "PREEMPTED"):
                stats['finished'] += 1
            stats['max_cpus'] = max(stats['max_cpus'], cpus)
            stats['max_gpus'] = max(stats['max_gpus'], gpus)
            stats['max_nodes'] = max(stats['max_nodes'], nodes)

            if start is not None and submit is not None and state not in UNSTARTED_STATES:
                stats['wait_total'] += max(0.0, start - submit)
                stats['wait_count'] += 1

            if start is None or state in UNSTARTED_STATES:
                continue  # never ran: no hours to attribute

            if self.is_stale(row['JobID'], state, end):
                stats['stale'] += 1
                self.stale += 1
                continue  # no defensible end time, so charge nothing

            # a running job has no end yet; bound it at now
            job_end = end if end is not None else max(start, now)
            if job_end <= start:
                job_end = start + elapsed

            # A job count is not divisible across periods the way hours are, so
            # it lands in the bucket the job started in.
            for label, b_start, b_end in self.buckets:
                if b_start <= start < b_end:
                    self.job_series[key][label] += 1
                    break

            # spread the job's resource-seconds over the buckets it overlaps,
            # clipped to the reporting window
            for label, b_start, b_end in self.buckets:
                overlap = min(job_end, b_end, window_end) - max(start, b_start, window_start)
                if overlap <= 0:
                    continue
                hours = overlap / 3600.0
                if cpus:
                    self.series[key][label] += cpus * hours
                    stats['cpu_hours'] += cpus * hours
                if gpus:
                    self.gpu_series[key][label] += gpus * hours
                    stats['gpu_hours'] += gpus * hours

    # --------------------------------------------------------------- ascii

    MEASURES = {
        "cpu": ("series", "CPU-hours", "{:,.0f}"),
        "gpu": ("gpu_series", "GPU-hours", "{:,.0f}"),
        "jobs": ("job_series", "jobs started", "{:,.0f}"),
    }

    def ascii_chart(self, measure="cpu", top=6, height=14, width=None, colour=True):
        """Stacked bar chart of a measure over time, drawn in the terminal.

        Columns are time buckets left to right; each column is stacked by
        entity. Segment heights are apportioned by largest remainder so the
        pieces always sum to the drawn bar height, and the top of each bar uses
        an eighth-block so totals resolve below one row.
        """
        attr, measure_label, fmt = self.MEASURES.get(measure, self.MEASURES["cpu"])
        series = getattr(self, attr)
        labels = [b[0] for b in self.buckets]
        if not labels:
            return ""

        ranked = [s for s in self.ranked()
                  if any(series[s['user']].get(l, 0) for l in labels)]
        if not ranked:
            return f"No {measure_label} recorded in this window."

        named = ranked[:top]
        rest = ranked[top:]
        totals = [sum(series[s['user']].get(l, 0.0) for s in ranked) for l in labels]
        peak = max(totals)
        if peak <= 0:
            return f"No {measure_label} recorded in this window."

        # y-axis label column
        ticks = {0: 0.0, height: peak}
        for frac in (0.25, 0.5, 0.75):
            ticks[int(round(height * frac))] = peak * frac
        label_w = max(len(fmt.format(v)) for v in ticks.values())

        if width is None:
            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 100
        room = max(20, width - label_w - 3)
        col_w = min(7, room // len(labels))
        if col_w < 1:
            return f"({len(labels)} periods need a wider terminal)"
        # only spend a column on the gap between bars once there is room for it
        bar_w = col_w - 1 if col_w >= 3 else col_w

        # stack each column
        columns = []
        for idx, label in enumerate(labels):
            exact = totals[idx] / peak * height
            full = int(exact)
            frac_eighths = int(round((exact - full) * 8))
            if frac_eighths == 8:
                full, frac_eighths = full + 1, 0

            values = [series[s['user']].get(label, 0.0) for s in named]
            if rest:
                values.append(sum(series[s['user']].get(label, 0.0) for s in rest))
            column_total = sum(values)
            if column_total <= 0:
                columns.append(([], 0, 0))
                continue

            # largest remainder, so segments sum to exactly `full` cells
            raw = [v / column_total * full for v in values]
            cells = [int(r) for r in raw]
            leftover = full - sum(cells)
            order = sorted(range(len(raw)), key=lambda i: raw[i] - cells[i], reverse=True)
            for i in order[:leftover]:
                cells[i] += 1
            top_slot = next((i for i in reversed(range(len(cells))) if cells[i] > 0),
                            next((i for i, v in enumerate(values) if v > 0), 0))
            columns.append((cells, frac_eighths, top_slot))

        def is_other(slot):
            return bool(rest) and slot == len(named)

        def fill(slot, cells, partial=None):
            """`cells` characters of this slot's fill, coloured or textured."""
            if not colour:
                glyph = (TERM_OTHER_GLYPH if is_other(slot)
                         else TERM_GLYPHS[slot % len(TERM_GLYPHS)])
                return glyph * cells
            body = (partial or "█") * cells
            code = TERM_OTHER if is_other(slot) else TERM_COLOURS[slot % len(TERM_COLOURS)]
            return f"\033[38;5;{code}m{body}{TERM_RESET}"

        lines = []
        for row in range(height, 0, -1):
            axis_label = fmt.format(ticks[row]).rjust(label_w) if row in ticks else " " * label_w
            mark = "┤" if row in ticks else "│"
            body = []
            for cells, frac_eighths, top_slot in columns:
                if not cells:
                    body.append(" " * col_w)
                    continue
                filled = sum(cells)
                if row <= filled:
                    # which series owns this row, counting from the bottom
                    running = 0
                    slot = 0
                    for i, c in enumerate(cells):
                        running += c
                        if row <= running:
                            slot = i
                            break
                    body.append(fill(slot, bar_w) + " " * (col_w - bar_w))
                elif row == filled + 1 and frac_eighths:
                    # partial top cell only reads as partial in colour mode; in
                    # texture mode the glyph itself has to carry identity
                    body.append(fill(top_slot, bar_w, PARTIAL_BLOCKS[frac_eighths])
                                + " " * (col_w - bar_w))
                else:
                    body.append(" " * col_w)
            lines.append(f"{axis_label} {mark}{''.join(body)}")

        lines.append(f"{fmt.format(0).rjust(label_w)} └{'─' * (col_w * len(labels))}")
        lines.extend(self._x_axis(labels, col_w, label_w))
        return "\n".join(lines)

    @staticmethod
    def _x_axis(labels, col_w, label_w):
        """Tick labels under the bars, thinned to whatever fits.

        Labels are written into a fixed-width buffer rather than padded per
        column: a label wider than its own column would otherwise push the line
        past the plot and overflow the terminal.
        """
        pad = " " * (label_w + 2)
        span = col_w * len(labels)

        def place(buf, pos, text):
            """Write text at pos if it fits and does not touch its neighbour."""
            if pos < 0 or pos + len(text) > span:
                return False
            if any(c != " " for c in buf[max(0, pos - 1):pos + len(text) + 1]):
                return False
            buf[pos:pos + len(text)] = list(text)
            return True

        if col_w >= max(len(l) for l in labels) + 1:      # full labels fit
            buf = [" "] * span
            for i, label in enumerate(labels):
                place(buf, i * col_w, label)
            return [pad + "".join(buf).rstrip()]

        # otherwise the leading token (day number, or month name) under each bar,
        # thinned so a label plus a space always fits, and a second line naming
        # the period whenever it changes
        heads = [l.split()[0] for l in labels]
        tails = [l.split()[1] if len(l.split()) > 1 else "" for l in labels]
        step = max(1, ceil((max(len(h) for h in heads) + 1) / col_w))

        buf = [" "] * span
        for i in range(0, len(labels), step):
            place(buf, i * col_w, heads[i])
        out = [pad + "".join(buf).rstrip()]

        if any(tails):
            buf2 = [" "] * span
            previous = None
            for i, tail in enumerate(tails):
                if tail and tail != previous:
                    place(buf2, i * col_w, tail)
                previous = tail
            if "".join(buf2).strip():
                out.append(pad + "".join(buf2).rstrip())
        return out

    def ascii_legend(self, measure="cpu", top=6, colour=True, width=None):
        attr, _, _ = self.MEASURES.get(measure, self.MEASURES["cpu"])
        series = getattr(self, attr)
        labels = [b[0] for b in self.buckets]
        ranked = [s for s in self.ranked()
                  if any(series[s['user']].get(l, 0) for l in labels)]
        if not ranked:
            return ""
        named, rest = ranked[:top], ranked[top:]

        chips = []
        for i, stats in enumerate(named):
            swatch = (f"\033[38;5;{TERM_COLOURS[i % len(TERM_COLOURS)]}m██{TERM_RESET}"
                      if colour else TERM_GLYPHS[i % len(TERM_GLYPHS)] * 2)
            chips.append(f"{swatch} {stats['user']}")
        if rest:
            swatch = (f"\033[38;5;{TERM_OTHER}m██{TERM_RESET}"
                      if colour else TERM_OTHER_GLYPH * 2)
            chips.append(f"{swatch} other ({len(rest)})")
        return "   ".join(chips)

    def ranked(self):
        """Users ordered by CPU-hours, then GPU-hours, descending."""
        return sorted(self.users.values(),
                      key=lambda s: (s['cpu_hours'], s['gpu_hours'], s['jobs']),
                      reverse=True)

    def has_gpu_usage(self):
        return any(s['gpu_hours'] > 0 for s in self.users.values())

    # ------------------------------------------------------------------ table

    def summary_table(self, top=None, colour=True):
        ranked = self.ranked()
        if not ranked:
            return "No usage found for this window."

        shown = ranked[:top] if top else ranked
        me = getpass.getuser()
        by = getattr(self, "by", "user") or "user"

        # the leading column names whatever a row represents; an aggregate row
        # gains a member count and drops the redundant agency column
        name_head = {"user": "USER", "group": "GROUP", "agency": "AGENCY"}[by]
        name_w = max(len(name_head), max(len(s['user']) for s in shown)) + 1
        second = "AGENCY" if by == "group" else ("" if by == "agency" else "AGENCY")
        second_w = (max(len(second), max(len(s['agency']) for s in shown)) + 1
                    if second else 0)

        head = f"{name_head:<{name_w}}"
        if second:
            head += f"{second:<{second_w}}"
        if by != "user":
            head += f"{'USERS':>7}"
        head += (f"{'JOBS':>8}{'CPU_HRS':>12}{'GPU_HRS':>10}{'AVG_WAIT':>10}"
                 f"{'MAX_JOB':>9}{'FAIL%':>7}")
        lines = [head, "-" * len(head)]

        for stats in shown:
            wait = (stats['wait_total'] / stats['wait_count']
                    if stats['wait_count'] else 0.0)
            fail = (100.0 * stats['failed'] / stats['finished']
                    if stats['finished'] else 0.0)
            biggest = f"{stats['max_cpus']}c"
            if stats['max_gpus']:
                biggest += f"/{stats['max_gpus']}g"
            row = f"{stats['user']:<{name_w}}"
            if second:
                row += f"{stats['agency']:<{second_w}}"
            if by != "user":
                row += f"{len(stats['members']):>7d}"
            row += (f"{stats['jobs']:>8d}{stats['cpu_hours']:>12,.0f}"
                    f"{stats['gpu_hours']:>10,.0f}{format_duration(wait):>10}"
                    f"{biggest:>9}{fail:>6.1f}%")
            if colour and by == "user" and stats['user'] == me:
                row = f"\033[1m{row}\033[0m"
            elif colour and fail >= 25.0 and stats['finished'] >= 10:
                # a quarter of finished work being lost is worth the eye
                row = f"\033[33m{row}\033[0m"
            lines.append(row)

        lead_w = name_w + second_w + (7 if by != "user" else 0)
        if top and len(ranked) > top:
            rest = ranked[top:]
            lines.append("-" * len(head))
            lines.append(f"{'+' + str(len(rest)) + ' others':<{lead_w}}"
                         f"{sum(s['jobs'] for s in rest):>8d}"
                         f"{sum(s['cpu_hours'] for s in rest):>12,.0f}"
                         f"{sum(s['gpu_hours'] for s in rest):>10,.0f}")

        total_cpu = sum(s['cpu_hours'] for s in ranked)
        total_gpu = sum(s['gpu_hours'] for s in ranked)
        lines.append("-" * len(head))
        lines.append(f"{'TOTAL':<{lead_w}}"
                     f"{sum(s['jobs'] for s in ranked):>8d}"
                     f"{total_cpu:>12,.0f}{total_gpu:>10,.0f}")
        return "\n".join(lines)

    def header(self):
        span = f"{self.start_time:%d %b %Y} to {self.end_time:%d %b %Y}"
        by = getattr(self, "by", "user") or "user"
        scope = ""
        if getattr(self, "group_filter", None):
            scope = " ".join(sorted(self.group_filter)) + " -- "
        elif getattr(self, "users_filter", None):
            scope = " ".join(sorted(self.users_filter)) + " -- "
        bits = [f"Usage by {by} -- {scope}{self.clusters} -- {span}"]
        if getattr(self, "partition", ""):
            bits.append(f"partition: {self.partition}")
        singular = {"user": "user", "group": "group", "agency": "agency"}[by]
        plural = {"user": "users", "group": "groups", "agency": "agencies"}[by]
        members = len({m for s in self.users.values() for m in s['members']})
        noun = singular if len(self.users) == 1 else plural
        count = f"{len(self.users)} {noun}"
        if by != "user":
            count += f" ({members} users)"
        # the count that matched the filter, not the count fetched from sacct
        matched = sum(s['jobs'] for s in self.users.values())
        allocations = f"{matched:,} job allocations"
        if matched != len(self.rows):
            allocations += f" of {len(self.rows):,}"
        bits.append(f"{count}, {allocations}, bucketed by {self.bucket_size}")
        if self.skipped:
            bits.append(f"({self.skipped} rows skipped: no user)")
        if self.stale:
            bits.append(
                f"\033[33mNote: {self.stale} job(s) are recorded as RUNNING but are no longer\033[0m\n"
                f"\033[33mknown to the controller -- no end time was ever written, so their\033[0m\n"
                f"\033[33melapsed time grows without bound. Their hours are excluded.\033[0m")
        return "\n".join(bits)

    # ------------------------------------------------------------------- plot

    def write_plot(self, path, top=8):
        """Stacked consumption over time. CPU and GPU get their own axes --
        two measures on one pair of scales would be unreadable and misleading."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        ranked = self.ranked()
        if not ranked:
            return False

        # Name the top users of *each* measure, not just CPU: the heaviest CPU
        # consumers are often not the GPU users at all, which would leave the
        # GPU panel as one undifferentiated "other" block.
        slots = len(SERIES_COLOURS)
        named = list(ranked[:top])
        if self.has_gpu_usage():
            by_gpu = sorted((s for s in ranked if s['gpu_hours'] > 0),
                            key=lambda s: s['gpu_hours'], reverse=True)
            for stats in by_gpu[:top]:
                if stats not in named and len(named) < slots:
                    named.append(stats)
        named = named[:slots]
        # Colour follows the user, fixed for the life of the figure, so the same
        # person is the same colour in both panels.
        colours = {s['user']: SERIES_COLOURS[i] for i, s in enumerate(named)}
        named_users = {s['user'] for s in named}
        rest = [s for s in ranked if s['user'] not in named_users]

        labels = [b[0] for b in self.buckets]
        show_gpu = self.has_gpu_usage()

        panels = [("CPU-hours", self.series)]
        if show_gpu:
            panels.append(("GPU-hours", self.gpu_series))

        fig, axes = plt.subplots(len(panels), 1, figsize=(max(9, len(labels) * 0.55), 4.4 * len(panels)),
                                 sharex=True, facecolor=SURFACE)
        if len(panels) == 1:
            axes = [axes]

        for ax, (measure, series) in zip(axes, panels):
            ax.set_facecolor(SURFACE)
            bottom = [0.0] * len(labels)
            for stats in named:
                values = [series[stats['user']].get(l, 0.0) for l in labels]
                if not any(values):
                    continue
                ax.bar(labels, values, bottom=bottom, label=stats['user'],
                       color=colours[stats['user']],
                       # surface-coloured gap keeps stacked segments legible
                       edgecolor=SURFACE, linewidth=1.0, width=0.78)
                bottom = [b + v for b, v in zip(bottom, values)]
            if rest:
                values = [sum(series[s['user']].get(l, 0.0) for s in rest) for l in labels]
                if any(values):
                    ax.bar(labels, values, bottom=bottom, label=f"other ({len(rest)})",
                           color=OTHER_COLOUR, edgecolor=SURFACE, linewidth=1.0, width=0.78)

            ax.set_ylabel(measure, color=INK_MUTED, fontsize=10)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.grid(axis='y', color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(GRID)
            ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)

        axes[0].set_title(self.header().split("\n")[0], color=INK,
                          fontsize=13, loc='left', pad=14)

        # One figure-level legend built from the colour map rather than from a
        # single axes' handles, so a GPU-only user that appears in the lower
        # panel is still identified. Always present for >=2 series, so identity
        # is never carried by colour alone.
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=colours[s['user']], label=s['user'])
                   for s in named
                   if s['cpu_hours'] > 0 or s['gpu_hours'] > 0]
        if rest:
            handles.append(Patch(facecolor=OTHER_COLOUR, label=f"other ({len(rest)})"))
        axes[0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1.005, 1.0),
                       frameon=False, fontsize=9, labelcolor=INK_MUTED, title="user",
                       title_fontproperties={'size': 9, 'weight': 'bold'})
        for label in axes[-1].get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        fig.tight_layout()
        fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches='tight')
        plt.close(fig)
        return True


def format_duration(seconds):
    seconds = int(max(0, seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def main():
    from .cluster_stat import parse_args

    args = parse_args()
    args_dict = vars(args)

    users_filter = None
    if args_dict.get('users'):
        users_filter = {u.strip() for u in args_dict['users'].split(",") if u.strip()}
    args_dict['users_filter'] = users_filter

    group_filter = None
    if args_dict.get('group'):
        # matched case-insensitively against both the full group and the agency
        group_filter = {g.strip().lower() for g in args_dict['group'].split(",") if g.strip()}
    args_dict['group_filter'] = group_filter

    history = UsageHistory(**args_dict)
    try:
        history.fetch()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    history.process()

    colour = sys.stdout.isatty()
    print(history.header())
    print()
    if not history.rows:
        # sacct exits 0 with no output for an unknown cluster as well as for a
        # genuinely empty window, so do not claim to know which happened.
        print(f"sacct returned no records for '{history.clusters}' over this window.\n"
              f"Check the cluster name (sacctmgr show cluster) and the -S/-E dates.")
        return
    if not history.users:
        print("Nothing matched that filter. "
              "Try --by group to see which groups exist.")
        return

    # A narrowed report is a focused one, so draw the chart without being asked.
    narrowed = bool(users_filter or group_filter or args_dict.get('by') != 'user')
    show_chart = (args_dict.get('chart') or narrowed) and not args_dict.get('no_chart')

    if show_chart:
        measure = args_dict.get('measure') or 'cpu'
        # the chart can resolve at most len(TERM_COLOURS) stacked series, but a
        # smaller --top should still narrow it so table and chart agree
        chart_top = min(args_dict.get('top') or len(TERM_COLOURS), len(TERM_COLOURS))
        chart = history.ascii_chart(measure=measure, top=chart_top, colour=colour)
        if chart:
            print(chart)
            legend = history.ascii_legend(measure=measure, top=chart_top, colour=colour)
            if legend:
                print("\n  " + legend)
            print()

    print(history.summary_table(top=args_dict.get('top'), colour=colour))

    # -o was given explicitly, or --plot asked for the default location
    wants_plot = args_dict.get('plot') or ('-o' in sys.argv or '--output' in sys.argv)
    if wants_plot:
        path = args_dict['output']
        if history.write_plot(path, top=args_dict.get('top') or 8):
            print(f"\nPlot saved to {path}.")
        else:
            print("\nNothing to plot for this window.", file=sys.stderr)


if __name__ == '__main__':
    main()
