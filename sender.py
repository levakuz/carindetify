from requests import post

url = 'http://195.2.74.79:8001/carident/by/2021-03'
# files = {'file': open('img.jpg', 'rb')}
# foo = post('http://0.0.0.0:8001/caident/by/2021.01', files=files)
files = {'file': open('1440.jpg', 'rb')}
foo = post(url, files=files, verify=False)
# x = requests.post(url, data = json.dumps(myobj))