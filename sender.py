from requests import post

url = 'http://192.168.1.105:8001/carident/ru/2020.01'
# files = {'file': open('img.jpg', 'rb')}
# foo = post('http://0.0.0.0:8001/caident/by/2021.01', files=files)
files = {'file': open('d32ea95.jpg', 'rb')}
foo = post(url, files=files, verify=False)
# x = requests.post(url, data = json.dumps(myobj))