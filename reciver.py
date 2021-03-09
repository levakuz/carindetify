from flask import jsonify, make_response, render_template, Flask, request
from waitress import serve
from main import read_and_image
import os
import base64
import time
import threading

app = Flask(__name__)

cont = ''
file = ''

@app.route('/carident/<country>/', methods=['POST'])
def carident(country):
    global cont
    global file
    cont = country
    print('re')
    start = time.time()
    image = request.files['file']
    print(image.filename)
    print(image)
    image.save(os.path.join('./', image.filename))
    file = './' + image.filename

    car_number = read_and_image('./' + image.filename, country)
    print(car_number)
    if car_number:
        response = {'status': '1', 'template': car_number}
        with open('./' + image.filename[:-4] + '.jpeg', mode='rb') as file:
            img = file.read()
        response['img'] = base64.encodebytes(img).decode('utf-8')
        os.remove('./' + image.filename[:-4] + '.jpeg')
        end = time.time()
        print(end - start)
        return jsonify(response)
    else:
        response = {'status': '0'}
        os.remove('./' + image.filename[:-4] + '.jpeg')
        end = time.time()
        print(end - start)
        return jsonify(response)

th1 = threading.Thread(target=serve(app, host='0.0.0.0', port=8001))
th2 = threading.Thread(target=carident, args=(cont, file), deamon=True)
th1.start()
th2.start()
th1.join()
th2.join()



#serve(app, host='0.0.0.0', port=8001)
