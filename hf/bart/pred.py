import os
import sys
from tqdm.auto import tqdm

from transformers import BartTokenizer
from torch.utils.data import DataLoader

from multitask_bart import BartForMultitaskLearning
from dataset import MultitaskDataset

tasks = sys.argv[1:]

model = BartForMultitaskLearning.from_pretrained(
    os.path.join('model', ''.join(task[0] for task in tasks))
).cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

dataset = MultitaskDataset(['response'], tokenizer, 'data', 'test', 256)
loader = DataLoader(dataset, batch_size=32)

outputs = []

for batch in tqdm(loader):
    outs = model.generate(
        input_ids=batch['source_ids'].cuda(),
        attention_mask=batch['source_mask'].cuda(), 
        max_length=256,
        num_beams=5,
        task=batch['task'][0]
    )

    for ids in outs:
        print(tokenizer.decode(ids, skip_special_tokens=True))
