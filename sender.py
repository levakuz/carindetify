from requests import post
import base64
url = 'http://192.168.1.105:8001/carident/ru/'
# files = {'file': open('img.jpg', 'rb')}
# foo = post('http://0.0.0.0:8001/caident/by/2021.01', files=files)
files = {'file': open('d32ea95.jpg', 'rb')}
foo = post(url, files=files, verify=False)
print(foo.json())
new_json = foo.json()
imgdata = base64.b64decode(new_json['img'])
filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(imgdata)
# x = requests.post(url, data = json.dumps(myobj))