from requests import post

url = 'http://192.168.1.105:8001/carident/ru/test2'
# files = {'file': open('img.jpg', 'rb')}
# foo = post('http://0.0.0.0:8001/caident/by/2021.01', files=files)
files = {'file': open('a7d5685.jpg', 'rb')}
foo = post(url, files=files, verify=False)
# x = requests.post(url, data = json.dumps(myobj))