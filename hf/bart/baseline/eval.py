import textwrap
from tqdm.auto import tqdm

from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from dataset import ResponseDataset

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained(
    'bart_large_response'
).cuda()

# visualize few predictions on test dataset
dataset = ResponseDataset(tokenizer, 'response_data', 'test', max_len=256)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

it = iter(loader)

batch = next(it)
print(batch['source_ids'].shape)

outs = model.generate(
    input_ids=batch['source_ids'].cuda(),
    attention_mask=batch['source_mask'].cuda(),
    max_length=256,
    min_length=16,
    num_beams=5,
)

dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

texts = [
    tokenizer.decode(ids, skip_special_tokens=True)
    for ids in batch['source_ids']
]
targets = [
    tokenizer.decode(ids, skip_special_tokens=True)
    for ids in batch['target_ids']
]

for i in range(32):
    lines = textwrap.wrap('Utterance:\n{}\n'.format(texts[i]))
    print('\n'.join(lines))

    lines = textwrap.wrap('Actual response:\n{}\n'.format(targets[i]))
    print('\n' + '\n'.join(lines))
    lines = textwrap.wrap('Predicted response:\n{}\n'.format(dec[i]))
    print('\n'.join(lines))

    print('=' * 70 + '\n')

# predict on all the test dataset
loader = DataLoader(dataset, batch_size=32, num_workers=4)

model.eval()

list_of_references = []
hypotheses = []

for batch in tqdm(loader):
    outs = model.generate(
        input_ids=batch['source_ids'].cuda(),
        attention_mask=batch['source_mask'].cuda(),
        max_length=256,
        min_length=16,
        num_beams=5,
    )

    hyps = [
        tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        for ids in outs
    ]
    refs = [
        [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)]
        for ids in batch['target_ids']
    ]

    hypotheses.extend(hyps)
    list_of_references.extend(refs)

print('BLEU score:', corpus_bleu(list_of_references, hypotheses))

print()
for i in range(4):
    weights = tuple(float(j == i) for j in range(4))
    print(
        str(i + 1) + '-gram:',
        corpus_bleu(list_of_references, hypotheses, weights)
    )
