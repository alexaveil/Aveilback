import os
from flask import Flask, jsonify, request
from inference import T5Model

app = Flask(__name__)
model = T5Model("model")

@app.route('/answer', methods=["POST"])
def classify_review():
    #Check for json input on request
    if not request.is_json:
        return jsonify(code=403, message="Bad request")
    #Check for valid data inside json
    request_data = request.get_json()
    if request_data is None:
        return jsonify(code=403, message="Bad request")
    #Check for required data in json content
    for key in ["api_key","question","interests"]:
        if key not in request_data.keys():
            return jsonify(code=403, message="Bad request")
    #Get fields
    question = request_data['question']
    interests = request_data['interests']
    api_key = request_data['api_key']
    #Check API Key: TODO:Change this
    if (api_key != "MyCustomerApiKey") or not isinstance(interests, list) or not isinstance(question, str):
        return jsonify(code=403, message="Bad request")
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
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
