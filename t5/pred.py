import os
import sys
from tqdm.auto import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader

from dataset import MultitaskDataset

tasks = sys.argv[1:]

model = T5ForConditionalGeneration.from_pretrained(
    os.path.join('model', ''.join(task[0] for task in tasks))
).cuda()
tokenizer = T5Tokenizer.from_pretrained('t5-base')

dataset = MultitaskDataset(['response'], tokenizer, '../data', 'test', 256)
loader = DataLoader(dataset, batch_size=32)

model.eval()
outputs = []

for batch in tqdm(loader):
    outs = model.generate(
        input_ids=batch['source_ids'].cuda(),
        attention_mask=batch['source_mask'].cuda(),
        max_length=256,
        num_beams=4,
        early_stopping=True,
        length_penalty=0.6
    )

    dec = [tokenizer.decode(ids) for ids in outs]
    outputs.extend(dec)

for output in outputs:
    print(output)
