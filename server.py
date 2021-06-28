import argparse
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
import gevent
import time
import logging
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, unset_jwt_cookies, set_access_cookies, get_jwt, get_jwt_identity, unset_access_cookies
from database.db_ops import find_user, register_user, update_interests
import bcrypt
from datetime import timedelta
from datetime import timezone
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from validation_utils import *
from datetime import datetime
import ast

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
jwt = JWTManager(app)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["30 per minute", "1 per second"])
#avoid limiting options requests
limiter.request_filter(lambda: request.method.upper() == 'OPTIONS')
# JWT Config
app.config["JWT_COOKIE_SECURE"] = True
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_SECRET_KEY"] = "wzdSFHpyIhRCZOkWPTsiYT94EfXcW3KjYU898JmvDkU1i87Ipf4RrDaFMU1J"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
app.config["SESSION_COOKIE_SAMESITE"]="Strict"

# Using an `after_request` callback, we refresh any token that is within 30
# minutes of expiring. Change the timedeltas to match the needs of your application.
@app.after_request
def refresh_expiring_jwts(response):
    try:
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
        if target_timestamp > exp_timestamp:
            access_token = create_access_token(identity=get_jwt_identity())
            set_access_cookies(response, access_token)
        return response
    except (RuntimeError, KeyError):
        # Case where there is not a valid JWT. Just return the original respone
        return response

@app.route("/register", methods=["POST"])
@limiter.limit("10/minute", override_defaults=False)
def register():
    #Avoid missing fields
    for field in ["first_name","last_name","password","birth_date", "email"]:
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
        return jsonify(message="Couldn't access DB"), 500
    #Create user
    hashed_pass = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())
    logger.info(hashed_pass)
    user_info = dict(first_name=first_name, last_name=last_name, email=email, password=hashed_pass, birth_date=birth_date)
    register_user(user_info)
    access_token = create_access_token(identity=email)
    response = jsonify(message="User added sucessfully")
    set_access_cookies(response, access_token)
    return response, 201

@app.route("/login", methods=["POST"])
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
    except Exception as e:
        return jsonify(message="Couldn't access DB"), 500
    #Check password
    user_pass = user["password"]
    if bcrypt.checkpw(password.encode('utf8'), user_pass):
        access_token = create_access_token(identity=email)
        response = jsonify(message="Login Succeeded", access_token=access_token)
        set_access_cookies(response, access_token)
        return response, 200
    else:
        return jsonify(message="Bad Email or Password"), 401

# Protect a route with jwt_required, which will kick out requests
# without a valid JWT present.
@app.route("/add_interests", methods=["POST"])
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
    # Access the identity of the current user with get_jwt_identity
    email = get_jwt_identity()
    exists = find_user({"email": email})
    #If user doesnt exist
    if not exists:
        return jsonify(message="User from session not found on db"), 401
    #Update interests to db
    try:
        update_interests(email, interests)
        return jsonify(message="User interests updated"), 200
    except Exception as e:
        logger.warning(e)
        return jsonify(message="There was an error updating the interests to the database"), 500

#Get user information
@app.route("/user_info", methods=["GET"])
@jwt_required()
def get_user_info():
    email = get_jwt_identity()
    user = find_user({"email": email})
    if not user:
        return jsonify(message="User from session not found on db"), 401
    else:
        #Delete unwanted fields
        del user["_id"]
        del user["password"]
        return jsonify(user), 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pipeline with steps')
    parser.add_argument('--run_local', action='store_true', help="If run local or not")
    parser.add_argument('--port', type=int, default=8000, help='Port for HTTP server (default: 8000)')
    parser.add_argument('--debug', action='store_true', help="If run with debugging or not")
    args = vars(parser.parse_args())
    ip = "localhost" if args["run_local"] else "0.0.0.0"
    port = args["port"]
    debug = args["debug"]
    # if debug
    if debug:
        app.debug = True
	# run app
    http_server = WSGIServer((ip, port), app)
    logger.info("Server started, running on {}:{}".format(ip, port))
    http_server.serve_forever()
