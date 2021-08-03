import os
from flask import Flask, jsonify, request
from inference import T5Model

app = Flask(__name__)
model = T5Model("model")

@app.route('/messages/ask_question', methods=["POST"])
def ask_question():
    #Check for json input on request
    if not request.is_json:
        return jsonify(code=403, message="Bad request, should use json")
    #Check for valid data inside json
    request_data = request.get_json()
    if request_data is None:
        return jsonify(code=403, message="Bad request, should use json")
    #Check for required data in json content
    for key in ["question","interests"]:
        if key not in request_data.keys():
            return jsonify(code=403, message="Missing {}".format(key))
    #Get fields
    question = request_data['question']
    interests = request_data['interests']
    if not isinstance(interests, list) or not isinstance(question, str):
        return jsonify(code=403, message="One of the parameters is incorrect, interests should be a list and question a string")
    #Ask the model
    try:
        answer = model.ask_question(interests, question)
        response = jsonify(message=answer)
        return response, 200
    except Exception as e:
        return jsonify(code=403, message="There was an error processing the request, please check your parameters")

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google Cloud
    # Run, a webserver process such as Gunicorn will serve the app.
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000), ssl_context='adhoc'))
