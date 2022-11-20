import json

from flask import Blueprint, Flask, request, Response
from flask_cors import CORS

from build_model import Model

ml = Blueprint('ml', __name__)
CORS(ml)

model = Model()

@ml.route('/predict', methods=['GET'])
def predict():
    if request.method != 'GET':
        return Response(
            json.dumps({'status': 'error', 'message': 'Method not allowed.'}),
            status=405
        )

    data = request.args

    # data = {
    #     "DATE": "2021-05-07",
    #     "OPEN_cur": 61.1850,
    #     "HIGH_cur": 61.1975,
    #     "LOW_cur": 61.1850,
    #     "CLOSE_cur": 61.1970,
    #     "VOL_cur": 36311.0
    # }
    res = model.predict(data)

    return Response(
        json.dumps({'status': 'ok', 'data': res}),
        status=200
    )    


if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(ml, url_prefix='')
    app.run(host='0.0.0.0', port=8080, debug=True)