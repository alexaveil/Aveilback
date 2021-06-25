import sys
import os
import ssl
from pymongo import MongoClient
sys.path.append(os.getcwd())
from config_parser import get_config_dict

#Get config
config = get_config_dict()
database_obj = config["database"]
uri = database_obj["uri"]
database = database_obj["database"]
client = MongoClient(
    uri,
    ssl=True,
    ssl_certfile='certificates/client.pem',
    ssl_cert_reqs=ssl.CERT_REQUIRED,
    ssl_ca_certs='certificates/ca.pem')
db = client[database]

#Define dbs
users_collection = db[database_obj["users"]]
conversations_collection = db[database_obj["conversations"]]
answer_collection = db[database_obj["answer_data"]]

x = users_collection.find()
print(list(x))
