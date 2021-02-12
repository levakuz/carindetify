from main import read_and_image
import os
import time
import glob, os
start = time.time()

parent_dir = './by/*'
number_tuple = []
folder = 'test1'
for pdf_file in glob.glob(os.path.join(parent_dir)):
    print (pdf_file)
    number = read_and_image(pdf_file, folder)
    number_tuple.append(number)
f = open('test.txt', "w")
for i in number_tuple:
    f.write(str(i))
    f.write("\n")
f.close()
end = time.time()
print(end-start)
print(len([name for name in os.listdir('./ru/') if os.path.isfile(os.path.join('./ru/', name))]))


