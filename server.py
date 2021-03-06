import argparse
from flask import Flask, request, jsonify, render_template
from gevent.pywsgi import WSGIServer
import gevent
import os
import time
import logging
from database.db_ops import find_user, register_user, update_interests, update_queston_count, add_question_with_answers, get_question_by_id, select_response, add_response_tagging_data, get_messages
from datetime import timedelta
from datetime import timezone
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from validation_utils import *
from datetime import datetime
import ast
from gpt3 import GPT3Handler
from bson.objectid import ObjectId
from questions.questions_list import question_list
from config_parser import get_config_dict
from flask_sslify import SSLify
import bcrypt
from flask_jwt_extended import (JWTManager, jwt_required, create_access_token, get_jwt_identity)
from flask_cors import CORS
import random
import requests
from requests.structures import CaseInsensitiveDict
from deploy.auth import get_cloud_run_token

#Init config to get certificates path
config_data = get_config_dict()

#Start logger
logger = logging.getLogger('Server')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

#Create server
app = Flask(__name__)
app.config['SECRET_KEY'] = "wzdSFHpyIhRCZOkWPTsiYT94EfXcW3KjYU898JmvDkU1i87Ipf4RrDaFMU1J"
app.config['JWT_SECRET_KEY'] = '8ej}s#;K(wPnavmkK`MHjT;3dw28ej}6s#;K(wPnavmkK`MHjT;3dw'
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)

#Initialize jwt manager
jwt = JWTManager(app)

#Protect to not use http
sslify = SSLify(app)

#Add cors
cors = CORS(app, resources={r"/*": {"origins": "*"}})

#Additional elements
gpt3_handler = GPT3Handler()
limiter = Limiter(app, key_func=get_remote_address, default_limits=["30 per minute", "1 per second"])
#avoid limiting options requests
limiter.request_filter(lambda: request.method.upper() == 'OPTIONS')
#Init variable for questions
MAX_QUESTIONS = 500

#Initialize swagger hash key
hashed_swagger_pass = b'$2b$12$IVjLwTjLBjxV7vjqTY4I2Ob2VqQd5GYlNckRPhM/t/TiFeZeyILUG'

@app.route("/user/register", methods=["POST"])
@limiter.limit("10/minute", override_defaults=False)
def register():
    #Avoid missing fields
    for field in ["first_name", "last_name", "password", "birth_date", "email"]:
        if field not in request.form.keys():
            return jsonify(message="Missing {}".format(field.replace("_"," "))), 400
    #Get fields
    email = request.form["email"]
    first_name = request.form["first_name"]
    last_name = request.form["last_name"]
    password = request.form["password"]
    birth_date = request.form["birth_date"]
    #Validate date
    if not valid_date(birth_date):
        return jsonify(message="Date is not valid, try with format dd/mm/yyyy"), 400
    #Validate email
    if not valid_email(email):
        return jsonify(message="Email is not valid"), 400
    #Get fields and check if user already created
    exists = find_user({"email": email})
    #Try to fetch user
    try:
        if exists:
            return jsonify(message="User Already Exist"), 409
    except Exception as e:
        logger.info(str(e))
        return jsonify(message="Couldn't access DB"), 500
    #Create user
    hashed_pass = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())
    user_info = dict(first_name=first_name, last_name=last_name, email=email, password=hashed_pass, birth_date=birth_date, question_count=0)
    try:
        user_entry = register_user(user_info)
        if user_entry is None:
            return jsonify(message="There was an error registering the user"), 500
    except Exception as e:
        return jsonify(message="There was an error registering the user"), 500
    #Login user
    user_id = user_entry.inserted_id
    return {"message":"User added sucessfully", "access_token": create_access_token(identity={"id": str(user_id)})}, 200

@app.route("/user/login", methods=["POST"])
def login():
    #Avoid missing fields
    for field in ["email","password"]:
        if field not in request.form.keys():
            return jsonify(message="Missing {}".format(field.replace("_"," "))), 400
    #Validate email
    email = request.form["email"]
    password = request.form["password"]
    if not valid_email(email):
        return jsonify(message="Email is not valid"), 400
    #Try to access db
    try:
        user = find_user({"email": email})
        if not user:
            return jsonify(message="There was an error in the credentials (email and password don't match)"), 400
    except Exception as e:
        return jsonify(message="Couldn't access DB"), 500
    #Check password
    user_pass = user["password"]
    if bcrypt.checkpw(password.encode('utf8'), user_pass):
        return {"message":"Logged in successfully", "access_token": create_access_token(identity={"id": str(user["_id"])})}, 200
    else:
        return jsonify(message="Bad Email or Password"), 401

# Protect a route with jwt_required, which will kick out requests
# without a valid JWT present.
@app.route("/user/add_interests", methods=["POST"])
@jwt_required()
def update_user_interests():
    #Check if interests provided
    if "interests" not in request.form:
        return jsonify(message="Missing interests"), 400
    #Validate interests type
    interests = request.form["interests"]
    if not isinstance(interests, list):
        try:
            interests = ast.literal_eval(interests)
        except Exception as e:
            return jsonify(message="Interests format is not valid"), 400
    #Validate interests content
    if not validate_interests(interests):
        return jsonify(message="Interests not valid, please check that the amount of interest is correct and that they are valid options"), 400
    #Update interests to db
    try:
        user_id = get_jwt_identity()["id"]
        update_interests(ObjectId(user_id), interests)
        return jsonify(message="User interests updated"), 200
    except Exception as e:
        logger.info(e)
        return jsonify(message="There was an error updating the interests to the database"), 500

#Get user information
@app.route("/user/user_info", methods=["GET"])
@jwt_required()
def get_user_info():
    user = find_user({"_id": ObjectId(get_jwt_identity()["id"])}, filter=["_id","password"])
    if not user:
        return jsonify(message="User from session not found on db"), 401
    else:
        #Delete unwanted fields
        return jsonify(user), 200

#Get question suggestions
@app.route("/messages/question_suggestion", methods=["GET"])
@jwt_required()
def question_suggestions():
    return jsonify(random.sample(question_list, 4)), 200

#Get question suggestions
@app.route("/messages/ask_custom_question", methods=["POST"])
@jwt_required()
def ask_custom_question():
    #TODO: Only have evaluation in this endpoint, not on cloud run
    #Check for json input on request
    if not request.is_json:
        return jsonify(message="Bad request, should use json"), 403
    #Check for valid data inside json
    request_data = request.get_json()
    if request_data is None:
        return jsonify(message="Bad request, should use json"), 403
    #Check for required data in json content
    for key in ["question","interests"]:
        if key not in request_data.keys():
            return jsonify(message="Missing {}".format(key)), 403
    #Get fields
    question = request_data['question']
    interests = request_data['interests']
    if not isinstance(interests, list) or not isinstance(question, str):
        return jsonify(message="One of the parameters is incorrect, interests should be a list and question a string"), 403
    #Get token for request
    try:
        endpoint_url = config_data["cloudrun"]["url"]+config_data["cloudrun"]["endpoint"]
        request_token = get_cloud_run_token(endpoint_url)
    except Exception as e:
        logger.info(e)
        return jsonify(message="There was an error getting the credentials for the request."), 500
    #Call cloud run
    try:
        headers = CaseInsensitiveDict()
        headers["Accept"] = "application/json"
        headers["Authorization"] = "Bearer {}".format(request_token)
        result = None
        result = requests.post(endpoint_url,
            json = {'question': question, 'interests': interests},
            headers = headers)
        return jsonify(result.json()), 200
    except Exception as e:
        logger.info(e)
        if result is not None:
            logger.info(result.text)
        return jsonify(message="There was an error getting the model response"), 500

#Ask GPT3 a question
@app.route("/messages/ask_question", methods=["POST"])
@jwt_required()
def ask_question():
    #Get question from request
    if not "question" in request.form:
        return jsonify(message="Question not provided"), 400
    question = request.form["question"]
    #Validate that it is a string
    if not isinstance(question, str):
        return jsonify(message="Question is not a string"), 400
    #Check that user has interests
    user_id = ObjectId(get_jwt_identity()["id"])
    user = find_user(user_id)
    if "interests" not in user:
        return jsonify(message="User doesn't have interests added"), 400
    #Check for question count
    if "question_count" not in user:
        return jsonify(message="Cannot get question count for user"), 400
    else:
        question_count = user["question_count"]
    #Check that question count is ok
    if question_count > MAX_QUESTIONS:
        return jsonify(message="Amount of questions performed exceeded"), 402
    #Ask GPT3 the question
    interests = user["interests"]
    try: #Ask 4 times
        responses = []
        for i in range(4):
            response = gpt3_handler.ask_interest_question(question, interests)
            responses.append(response)
    except Exception as e:
        return jsonify(message="There was an error generating the question response"), 500
    #Update question count (TODO: Update with tokenizer count)
    try:
        update_queston_count(user_id, 1) #This adds to question count, avoid overwriting between requests with wrong value
    except Exception as e:
        return jsonify(message="There was an error updating the response"), 500
    #Add question answer pair to find best collection, saving its
    #Update history of questions
    try:
        result_conversation = add_question_with_answers(user_id, question, responses)
        if result_conversation is None:
            return jsonify(message="Inserting response failed"), 500
    except Exception as e:
        return jsonify(message="There was an error saving the responses to the db"), 500
    #Update previous question so that it has a good value
    data = {"question_id": str(result_conversation.inserted_id), "responses":responses}
    return jsonify(data), 200

#Ask GPT3 a question
@app.route("/messages/select_question", methods=["POST"])
@jwt_required()
def select_question():
    #Get question_id and option_selected
    for param in ["question_id", "option_selected"]:
        if not param in request.form:
            return jsonify(message="{} not provided".format(param)), 400
    #Get parameters
    question_id = request.form["question_id"]
    option_selected = request.form["option_selected"]
    #Validate option selected
    if (not option_selected.isdigit()) or (not 1 <= int(option_selected) <= 4):
        return jsonify(message="Option selected is not valid, must be an integer between 1 and 4"), 400
    option_selected = int(option_selected)
    #Verify that the question corresponds to the user of the session, and that the question exists
    user_id = ObjectId(get_jwt_identity()["id"])
    user = find_user({"_id":user_id})
    question = get_question_by_id(question_id)
    #Handle conversation not found
    if question is None:
        return jsonify(message="Question with id not found, please check the id provided"), 400
    #Handle cases when response selected already set
    if 'option_selected' in question:
        return jsonify(message="Response already selected"), 400
    #Check if question assigned to user
    if question["user_id"] != user_id:
        return jsonify(message="Question provided not assigned to user"), 400
    #Assign response to conversation
    try:
        select_response(question_id, option_selected)
    except Exception as e:
        return jsonify(message="There was error updating the selected response"), 500
    #Add data to conversation
    try:
        #Iterate over data and add to db
        user_interests = user["interests"]
        data_to_insert = []
        answers = question["answers"]
        date = datetime.now()
        for i in range(4):
            index = i+1
            selected = index == option_selected
            data = {"user_interests":user_interests, "question":question["question"], "answer":answers[i],"user_id":user_id, "question_id":ObjectId(question_id), "selected":selected, "date": date}
            data_to_insert.append(data)
        add_response_tagging_data(data_to_insert)
    except Exception as e:
        return jsonify(message="There was error generating data from selected response"), 500
    #Get result
    return jsonify(message="Response updated"), 200

#Get recent conversation messages
@app.route("/messages/get_messages", methods=["GET"])
@jwt_required()
def get_messages_user():
    def clean_id(entry):
        id = entry["_id"]
        del entry["_id"]
        entry["question_id"]=str(id)
        return entry
    #Handle input
    page = request.args.get('page', None)
    if page is None:
        return jsonify(message="Page not provided"), 400
    #Try to pull messages
    try:
        user_id = ObjectId(get_jwt_identity()["id"])
        page = int(page)
        results_per_page = 10
        messages = get_messages(user_id, offset=results_per_page*page, limit_messages=results_per_page)
        messages = list(map(clean_id, messages))
        return jsonify(messages), 200
    except Exception as e:
        return jsonify(message="There was en error getting the messages"), 500

@app.route('/api/docs', methods=["GET"])
def get_docs():
    try:
        password = request.args.get('password', None)
        if password is None:
            return jsonify(message="Password not provided"), 400
        if bcrypt.checkpw(password.encode('utf8'), hashed_swagger_pass):
            return render_template('swaggerui.html')
        else:
            return jsonify(message="Unauthorized"), 401
    except Exception as e:
        return jsonify(message="There was an error processing your request"), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pipeline with steps')
    parser.add_argument('--run_local', action='store_true', help="If run local or not")
    parser.add_argument('--port', type=int, default=8000, help='Port for HTTP server (default: 8000)')
    parser.add_argument('--debug', action='store_true', help="If run with debugging or not")
    args = vars(parser.parse_args())
    ip = "localhost" if args["run_local"] else "0.0.0.0"
    port = args["port"]
    debug = args["debug"]
    certificates_folder = config_data["certificates"]["path"]
    # if debug
    if debug:
        app.debug = True
	# run app
    http_server = WSGIServer((ip, port), app, keyfile=os.path.join(certificates_folder, 'privkey.pem'), certfile=os.path.join(certificates_folder, 'fullchain.pem'))
    logger.info("Server started, running on {}:{}".format(ip, port))
    http_server.serve_forever()
