import sys
import os
import ast
sys.path.append(os.getcwd())
from transformers_pytorch.dataset_utils import get_dataset, make_logdir
from utils import split_filename
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

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader


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
    parser.add_argument('--batch_size', default=2, required=False, help="Batch size")
    parser.add_argument('--metrics', default=["f1"], required=False, help="Metrics to evaluate")
    args = parser.parse_args()
    train(args)















    #
