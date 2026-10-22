"""Extract final-trial timings and validation metrics from batch stdout into CSV.

Usage: python research/frontier_eigh/summarize.py output/custom-eigh/job-*.out
Native-half failures still have RESULT rows; consult stdout CHECK markers/stderr.
"""
import csv
from pathlib import Path
import re
import sys

rows = []
for path in map(Path, sys.argv[1:]):
    timing = {}
    binary = digest = command = ""
    for line in path.read_text().splitlines():
        if line.startswith("COMMAND "):
            command = line[len("COMMAND "):]
            timing = {}
        elif re.match(r"^[a-f0-9]{64}  ", line):
            digest, binary = line.split()
        elif line.startswith(("TIMING ", "RESULT ")):
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            if line.startswith("TIMING "):
                timing = fields
            else:
                rows.append(dict(job=path.stem[len("job-"):], binary=binary,
                                 sha256=digest, command=command, **{**timing, **fields}))
if not rows:
    raise SystemExit("No RESULT records found")
writer = csv.DictWriter(sys.stdout, list(dict.fromkeys(k for row in rows for k in row)))
writer.writeheader()
writer.writerows(rows)
