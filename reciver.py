from flask import jsonify, make_response, render_template, Flask, request
from waitress import serve
from main import read_and_image
import os

app = Flask(__name__)


@app.route('/carident/<country>/<date>', methods=['POST'])
def carident(country, date):
    print('re')
    image = request.files['file']
    print(image.filename)
    print(image)
    image.save(os.path.join('./' + date, image.filename))
    car_number = read_and_image('./' + date + '/' + image.filename, date + '/', country)
    if car_number:
        return car_number
    else:
        return 'False'


serve(app, host='0.0.0.0', port=8001)
