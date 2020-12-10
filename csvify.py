import csv
import sys

in_paths = sys.argv[1:-1]
out_path = sys.argv[-1]

cols = []
for in_path in in_paths:
    col = [in_path]
    with open(in_path) as f:
        for line in f:
            col.append(line.strip())
    cols.append(col)

rows = list(zip(*cols))
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
