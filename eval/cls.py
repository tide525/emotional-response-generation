import sys

from sklearn import metrics

target_path = sys.argv[1]
pred_path = sys.argv[2]

with open(target_path) as f:
    targets = [int(line) for line in f]

with open(pred_path) as f:
    preds = [int(line) for line in f]

print(metrics.classification_report(targets, preds, digits=4))
