from flask import jsonify, make_response, render_template, Flask, request
from waitress import serve
from main import read_and_image
import os
import requests
from os import path


app = Flask(__name__)


@app.route('/carident/<country>/<date>/<id>', methods=['POST'])
def carident(country, date, id):
    print('re')
    image = request.files['file']
    print(image.filename)
    print(image)
    if path.exists("./" + date) is False:
        os.mkdir("./" + date)
    image.save(os.path.join('./' + date, image.filename))
    car_number = read_and_image('./' + date + '/' + image.filename, date + '/', country)
    print(date)
    url = 'https://forsage.by/_utils/car_number_uploader.php'
    files = {'file': open('./' + date + '/' + image.filename[:-4] + '.jpeg', 'rb')}
    foo = requests.post(url, files=files, data={'date': date, 'number': car_number, 'id': id})
    os.remove('./' + date + '/' + image.filename)
    if car_number:
        os.remove('./' + date + '/' + image.filename[:-4] + '.jpeg')
        return car_number
    else:
        os.remove('./' + date + '/' + image.filename[:-4] + '.jpeg')
        return 'False'


serve(app, host='0.0.0.0', port=8001)
