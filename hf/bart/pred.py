import os
import sys
from tqdm.auto import tqdm

from transformers import BartTokenizer
from torch.utils.data import DataLoader

from multitask_bart import BartForMultitaskLearning
from dataset import MultitaskDataset

model_name = sys.argv[1]

model = BartForMultitaskLearning.from_pretrained(
    os.path.join('model', model_name)
).cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

dataset = MultitaskDataset(['response'], tokenizer, '../data', 'test', 256)
loader = DataLoader(dataset, batch_size=32)

model.eval()
outputs = []

for batch in tqdm(loader):
    outs = model.generate(
        input_ids=batch['source_ids'].cuda(),
        attention_mask=batch['source_mask'].cuda(), 
        max_length=256,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
        task=batch['task'][0]
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    outputs.extend(dec)

with open(os.path.join('pred', model_name) + '.txt', 'w') as f:
    for output in outputs:
        f.write(output + '\n')
