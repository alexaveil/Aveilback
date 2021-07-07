import sys
import os
import ast
sys.path.append(os.getcwd())
from utils import load_json, split_filename
from datasets import load_dataset
from itertools import chain
import argparse
import logging
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_scheduler
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pprint import pprint

logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_dataset(tokenizer, dataset_path, dataset_cache):
    #https://gist.github.com/thomwolf/ecc52ea728d29c9724320b38619bd6a6
    #https://www.kaggle.com/taffydas/huggingface-convai-gpt2-utils
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
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

def train(args):
    logger.info("Loading tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(args.model_type)

    logger.info("Loading dataset")
    dataset_folder, dataset_filename, dataset_extension = split_filename(args.dataset_path)
    cache_path = os.path.join(dataset_folder, dataset_filename+"_cache_"+type(tokenizer).__name__+dataset_extension) # To avoid using GPT cache for GPT-2 and vice-versa
    dataset = get_dataset(tokenizer, args.dataset_path, cache_path)

    logger.info("Getting dataloaders")
    small_train_dataset = dataset["train"]
    small_eval_dataset = dataset["train"]
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)

    logger.info("Loading model")
    model = T5ForConditionalGeneration.from_pretrained(args.model_type)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    logger.info("Loading metrics")
    if isinstance(args.metrics, str):
        metric_list = ast.literal_eval(args.metrics)
    else:
        metric_list = args.metrics
    metrics = {metric: load_metric(metric) for metric in metric_list}

    logger.info("Starting training")
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(args.num_epochs):
        #Traing for an epoch
        model.train()
        for batch in train_dataloader:
            utter = batch["utterances"][0]
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
        #Evaluate
        logger.info("Evaluating")
        model.eval()
        for batch in eval_dataloader:
            utter = batch["utterances"][0]
            history = utter["history"][0][0].to(device)
            candidate = utter["candidates"][0][0].to(device)
            with torch.no_grad():
                outputs = model(input_ids=history, labels=candidate)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            for _, metric in metrics.items():
                metric.add_batch(predictions=predictions[0], references=candidate[0])
        current_metrics = {}
        for metric_name, metric in metrics.items():
            if metric_name == "f1":
                score = metric.compute(average="micro")
            else:
                score = metric.compute()
            current_metrics.update(score)
        logger.info(str(current_metrics))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pipeline with steps')
    parser.add_argument('--checkpoint_folder', default="checkpoint", required=False, help="Where to save the models generated")
    parser.add_argument('--log_folder', default="logs", required=False, help="Where to save the logs")
    parser.add_argument('--dataset_path', default="TEST_DATASET/persona_example.json", required=False, help="Dataset path")
    parser.add_argument('--model_type', default="t5-small", required=False, help="Model type")
    parser.add_argument('--num_epochs', default=450, required=False, help="Amount of epochs")
    parser.add_argument('--learning_rate', default=5e-5, required=False, help="Learning rate")
    parser.add_argument('--batch_size', default=1, required=False, help="Batch size")
    parser.add_argument('--metrics', default=["f1"], required=False, help="Metrics to evaluate")
    args = parser.parse_args()
    train(args)















    #
