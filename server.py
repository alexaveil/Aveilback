import argparse
from flask import Flask
from gevent.pywsgi import WSGIServer
import gevent
import time
import logging

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

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pipeline with steps')
    parser.add_argument('--run_local', action='store_true', help="If run local or not")
    parser.add_argument('--port', type=int, default=8000, help='Port for HTTP server (default: 8000)')
    args = vars(parser.parse_args())
    ip = "localhost" if args["run_local"] else "0.0.0.0"
    port = args["port"]
	# run app
    http_server = WSGIServer((ip, port), app)
    logger.info("Server started, running on {}:{}".format(ip, port))
    http_server.serve_forever()
