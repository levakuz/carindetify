from flask import jsonify, make_response, render_template, Flask, request
from waitress import serve
from main import read_and_image
import os
import base64



app = Flask(__name__)


@app.route('/carident/<country>/', methods=['POST'])
def carident(country):
    print('re')
    image = request.files['file']
    print(image.filename)
    print(image)
    image.save(os.path.join('./', image.filename))
    car_number = read_and_image('./' + image.filename, country)
    if car_number:
        response = {'status': '1', 'template': car_number}
        with open('./' + image.filename[:-4] + '.jpeg', mode='rb') as file:
            img = file.read()
        response['img'] = base64.encodebytes(img).decode('utf-8')
        os.remove('./' + image.filename[:-4] + '.jpeg')
        return jsonify(response)
    else:
        response = {'status': '0'}
        os.remove('./' + image.filename[:-4] + '.jpeg')
        return jsonify(response)


serve(app, host='0.0.0.0', port=8001)
