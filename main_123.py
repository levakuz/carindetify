import requests
import json
url = 'https://forsage.ru/_utils/car_number_uploader.php'
myobj = {'person': 'somevalue'}
files = {'file': open('11_d_850.jpeg', 'rb')}
foo = requests.post(url, files=files, verify=False)
# x = requests.post(url, data = json.dumps(myobj))