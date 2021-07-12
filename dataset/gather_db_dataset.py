import argparse
import sys
import os
import numpy as np
from datetime import datetime
import logging
sys.path.append(os.getcwd())
from utils import save_json, now2string
import ast
from database.db_ops import get_data
import random

logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def to_persona_chat(entry):
    """Changes the format provided on the DB to the persona chat single entry format"""
    interests = entry["user_interests"]
    question = entry["question"]
    answer = entry["answer"]
    return {
        "personality": ["user likes "+interest+"." for interest in interests],
        "utterances":[
            {
                "candidates":[answer],
                "history":[question]
            }
        ]
    }

def get_dataset(start_date, out_folder, proportions):
    """Get all the entries from the database from start date on and save a json file with the data generated in the format required for training"""
    #Make out dir and gather proportions
    os.makedirs(out_folder, exist_ok=True)
    proportions = ast.literal_eval("["+proportions+"]")
    #Get data from DB
    if start_date is not None:
        logger.info("Starting date is: {}".format(start_date))
        date_time_obj = datetime.strptime(date_time_str, '%d/%m/%y %H:%M:%S')
        data = get_data(from_time=date_time_obj)
    else:
        logger.info("No starting date added, getting all entries on DB")
        data = get_data()
    #Iterate over data and transform to persona chat format
    result = []
    for entry in data:
        if "selected" not in entry or not entry["selected"]:
            continue
        result.append(to_persona_chat(entry))
    #Split data in train, validation, test set
    random.seed(2021)
    random.shuffle(result)
    train_proportion = proportions[0]/100 # Percentage to float
    val_proportion = (proportions[1]/(100-proportions[0])) # Get total from remaining part and transform to float
    train, val_test = np.split(result, [int(train_proportion  * len(result))])
    valid, test = np.split(val_test, [int(val_proportion * len(val_test))])
    #Save dataset
    path_data = os.path.join(out_folder, "dataset_"+now2string(flatten=True)+".json")
    data = {"train":list(train),"valid":list(valid), "test":list(test)}
    save_json(data, path_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pipeline with steps')
    parser.add_argument('--start_date', required=False, help="From which date to start taking data, format: 18/09/19 01:55:19")
    parser.add_argument('--out_folder', required=True, help="Folder where to save dataset")
    parser.add_argument('--proportions', default="75,15,10", required=False, help="Splits proportion, format: train,valid,test")
    args = vars(parser.parse_args())
    start_date = args["start_date"]
    out_folder = args["out_folder"]
    proportions = args["proportions"]
    get_dataset(start_date, out_folder, proportions)
