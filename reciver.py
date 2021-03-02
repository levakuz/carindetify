from flask import jsonify, make_response, render_template, Flask, request
from waitress import serve
from main import read_and_image
import os
import requests

app = Flask(__name__)


@app.route('/carident/<country>/<date>', methods=['POST'])
def carident(country, date):
    print('re')
    image = request.files['file']
    print(image.filename)
    print(image)
    image.save(os.path.join('./' + date, image.filename))
    car_number = read_and_image('./' + date + '/' + image.filename, date + '/', country)
    url = 'https://forsage.by/_utils/car_number_uploader.php'
    files = {'file': open('./' + date + '/' + image.filename[:-4] + '.jpeg', 'rb')}
    foo = requests.post(url, files=files, data={'date': date, 'car_number': car_number})
    os.remove('./' + date + '/' + image.filename)
    if car_number:
        os.remove('./' + date + '/' + image.filename[:-4] + '.jpeg')
        return car_number
    else:
        os.remove('./' + date + '/' + image.filename[:-4] + '.jpeg')
        return 'False'


serve(app, host='0.0.0.0', port=8001)
