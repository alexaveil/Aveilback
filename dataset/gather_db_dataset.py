import argparse
import sys
import os
from datetime import datetime
import logging
sys.path.append(os.getcwd())
from utils import save_json
from database.db_ops import get_data

logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_dataset(start_date, out_folder):
    """Get all the entries from the database from start date on and save a json file with the data generated in the format required for training"""
    os.makedirs(out_folder, exist_ok=True)
    if start_date is not None:
        logger.info("Starting date is: {}".format(start_date))
        date_time_obj = datetime.strptime(date_time_str, '%d/%m/%y %H:%M:%S')
        data = get_data(from_time=date_time_obj)
    else:
        logger.info("No starting date added, getting all entries on DB")
        data = get_data()
    path_data = os.path.join(out_folder, "dataset.json")
    save_json(data, path_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pipeline with steps')
    parser.add_argument('--start_date', required=False, help="From which date to start taking data, format: 18/09/19 01:55:19")
    parser.add_argument('--out_folder', required=True, help="Folder where to save dataset")
    args = vars(parser.parse_args())
    start_date = args["start_date"]
    out_folder = args["out_folder"]
    get_dataset(start_date, out_folder)
