"""Extract final-trial phases, warm timing ranges, and validation metrics as CSV.

Usage: python3 research/frontier_eigh/summarize.py output/custom-eigh/job-*.out
Native-half failures still have RESULT rows; consult stdout CHECK markers/stderr.
"""
import csv
from pathlib import Path
import re
from statistics import median
import sys

rows = []
for path in map(Path, sys.argv[1:]):
    timing = {}
    trials = []
    binary = digest = command = ""
    for line in path.read_text().splitlines():
        if line.startswith("COMMAND "):
            command = line[len("COMMAND "):]
            timing = {}
            trials = []
        elif re.match(r"^[a-f0-9]{64}  ", line):
            digest, binary = line.split()
        elif line.startswith(("TIMING ", "RESULT ")):
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            if line.startswith("TIMING "):
                timing = fields
                trials.append(float(fields["total"]))
            else:
                warm = trials[1:]
                rows.append(dict(job=path.stem[len("job-"):], binary=binary,
                                 sha256=digest, command=command, trials=len(trials),
                                 cold_total=trials[0], warm_median=median(warm) if warm else "",
                                 warm_min=min(warm) if warm else "", warm_max=max(warm) if warm else "",
                                 **{**timing, **fields}))
if not rows:
    raise SystemExit("No RESULT records found")
writer = csv.DictWriter(sys.stdout, list(dict.fromkeys(k for row in rows for k in row)))
writer.writeheader()
writer.writerows(rows)
