import requests
import json
url = 'https://forsage.ru/_utils/car_number_uploader.php'
myobj = {'person': 'somevalue'}
files = {'file': open('a7d5685.jpg', 'rb')}
foo = requests.post(url, files=files, verify=False, data={'date': '2020.01'})
# foo = requests.post(url, verify=False, data={'number': '111', 'id':'747480'})
#x = requests.post(url, data = json.dumps(myobj))