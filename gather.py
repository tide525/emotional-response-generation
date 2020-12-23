import csv
import os
import sys

data_dir = sys.argv[1]
output_dirs = sys.argv[2:]

cols = []
for lang in ['source', 'target']:
    col = [lang]
    with open(os.path.join(data_dir, 'test.' + lang)) as f:
        for line in f:
            col.append(line.strip())
    cols.append(col)

for output_dir in output_dirs:
    col = [os.path.basename(os.path.dirname(output_dir))]
    with open(os.path.join(output_dir, 'pred.txt')) as f:
        for line in f:
            col.append(line.strip())
    cols.append(col)

rows = zip(*cols)  # transpose

with open('gather.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
