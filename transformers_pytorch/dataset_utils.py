import torch
from utils import load_json
import os
import logging

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

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir
