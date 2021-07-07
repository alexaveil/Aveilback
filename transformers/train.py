import sys
import os
sys.path.append(os.getcwd())
from utils import load_json
from datasets import load_dataset

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

tokenizer = T5Tokenizer.from_pretrained('t5-small')

print("Loaded dataset, tokenizing")
#https://gist.github.com/thomwolf/ecc52ea728d29c9724320b38619bd6a6
def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)

raw_dataset = load_json("TEST_DATASET/persona_example.json") #load_json('dataset/personachat_self_original.json')
tokenized_datasets = tokenize(raw_dataset)


def get_dataset(tokenizer, dataset_path, dataset_cache):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)

        torch.save(dataset, dataset_cache)
    return dataset








print("Splitting dataset")
small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["train"]
print(small_train_dataset)
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
model = T5ForConditionalGeneration.from_pretrained("t5-small")
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
print("Initializing model")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
progress_bar = tqdm(range(num_training_steps))

print("Training")
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        print(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
print("Evaluating")
metric = load_metric("f1")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
score = metric.compute()
print(score)
