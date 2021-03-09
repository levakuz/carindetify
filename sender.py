from requests import post
import base64
import time
url = 'http://192.168.0.33:8001/carident/ru/'
# files = {'file': open('img.jpg', 'rb')}
# foo = post('http://0.0.0.0:8001/caident/by/2021.01', files=files)
files = {'file': open('test2QWI.jpg', 'rb')}
start = time.time()
foo = post(url, files=files, verify=False)
end = time.time()
print(end - start)

new_json = foo.json()
imgdata = base64.b64decode(new_json['img'])
filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(imgdata)
# x = requests.post(url, data = json.dumps(myobj))