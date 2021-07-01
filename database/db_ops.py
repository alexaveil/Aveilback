import sys
import os
import ssl
from pymongo import MongoClient
sys.path.append(os.getcwd())
from config_parser import get_config_dict
from datetime import datetime
from bson.objectid import ObjectId

#Get config
config = get_config_dict()
database_obj = config["database"]
uri = database_obj["uri"]
database = database_obj["database"]

path_certificates=config["certificates"]["path"]
client = MongoClient(
    uri,
    ssl=True,
    ssl_certfile=os.path.join(path_certificates,'client.pem'),
    ssl_cert_reqs=ssl.CERT_REQUIRED,
    ssl_ca_certs=os.path.join(path_certificates,'ca.pem'))
db = client[database]

#Define dbs
users_collection = db[database_obj["users"]]
conversations_collection = db[database_obj["conversations"]]
responses_tagged_data = db[database_obj["responses_tagged_data"]]

#User operations
def find_user(user_dict, filter=[]):
    """Remove filter fields from result"""
    result = users_collection.find_one(user_dict)
    if result is not None:
        for field in filter:
            if field in result:
                del result[field]
    return result

def register_user(user_dict):
    return users_collection.insert_one(user_dict)

def update_interests(user_id, interests):
    users_collection.update_one({'_id': user_id},{'$set': {'interests': interests}}, upsert=False)

def update_queston_count(user_id, increase):
    users_collection.find_one_and_update({'_id': user_id}, {"$inc": {'question_count': increase}}, upsert=False)

#Conversations operations
def add_question_with_answers(user_id, question, answers):
    dict_insert = {"question":question, "answers":answers, "user_id": user_id, "date":datetime.now()}
    return conversations_collection.insert_one(dict_insert)

def get_question_by_id(question_id):
    return conversations_collection.find_one({"_id":ObjectId(question_id)})

def get_messages(user_id, offset=0, limit_messages=10, sort_order=-1): #Sort order -1 to set latest first
    return list(conversations_collection.find({"user_id":user_id},{'_id': False, "user_id":False}).sort([('date', sort_order)]).skip(offset).limit(limit_messages))

def select_response(question_id, option_selected):
    conversations_collection.update_one({"_id":ObjectId(question_id)}, {"$set": {'option_selected': option_selected}}, upsert=False)

#Data responses collection
def add_response_tagging_data(data_to_insert):
    responses_tagged_data.insert_many(data_to_insert)
