import os
from bs4 import BeautifulSoup

print("image_id,class,bounds")

directory = "./Annotations"

for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file_path:
        data = file_path.read()
    xml = BeautifulSoup(data, "xml")
    boxes = xml.find_all('bndbox')
    image_name = xml.find('filename').text.split('.')[0]
    for box in boxes:
        print("{},oil-well,\"({},{},{},{})\"".format(image_name, box.xmin.text, box.ymin.text, box.xmax.text, box.ymax.text))
