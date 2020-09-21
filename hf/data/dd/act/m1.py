import os
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]

for split in ['train', 'validation', 'test']:
    text_path = os.path.join(in_dir, split, 'dialogues_' + split + '.txt')
    act_path = os.path.join(in_dir, split, 'dialogues_act_' + split + '.txt')

    pairs = []
    with open(text_path) as text_f, open(act_path) as act_f:
        for text_line, act_line in zip(text_f, act_f):
            texts = text_line.split('__eou__')[:-1]
            acts = act_line.split()

            assert len(texts) == len(acts)

            for text, act in zip(texts, acts):
                pairs.append([text.strip(), int(act) - 1])
    
    if split == 'validation':
        split = 'val'

    for i, lang in enumerate(['source', 'target']):
        out_path = os.path.join(out_dir, split + '.' + lang)
        with open(out_path, 'w') as f:
            for j in range(len(pairs)):
                f.write(str(pairs[j][i]) + '\n')
