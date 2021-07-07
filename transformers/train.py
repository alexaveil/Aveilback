import sys
import os
sys.path.append(os.getcwd())
from utils import load_json
from datasets import load_dataset
from itertools import chain

import logging
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_scheduler
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pprint import pprint

logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

tokenizer = T5Tokenizer.from_pretrained('t5-small')

print("Loaded dataset, tokenizing")
#https://gist.github.com/thomwolf/ecc52ea728d29c9724320b38619bd6a6
#https://www.kaggle.com/taffydas/huggingface-convai-gpt2-utils

def get_dataset(tokenizer, dataset_path, dataset_cache):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        dataset = load_json(dataset_path)
        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                print(obj)
                return tokenizer(obj, return_tensors='pt', padding=True, truncation=True).input_ids
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset

dataset = get_dataset(tokenizer, "TEST_DATASET/persona_example.json", "TEST_DATASET/persona_example_cache.json")
print("Splitting dataset")
small_train_dataset = dataset["train"]
small_eval_dataset = dataset["train"]
pprint(small_train_dataset)
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=1)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=1)
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
        utter = batch["utterances"][0]
        pprint(utter["history"][0])
        history = utter["history"][0][0].to(device)
        candidate = utter["candidates"][0][0].to(device)
        #Get input_ids and labels from batch
        outputs = model(input_ids=history, labels=candidate)
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
    utter = batch["utterances"][0]
    history = utter["history"][0][0].to(device)
    pprint(history)
    candidate = utter["candidates"][0][0].to(device)
    pprint(candidate)
    with torch.no_grad():
        outputs = model(input_ids=history, labels=candidate)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    print(predictions)
    print(candidate)
    metric.add_batch(predictions=predictions, references=candidate)
score = metric.compute()
print(score)



#
