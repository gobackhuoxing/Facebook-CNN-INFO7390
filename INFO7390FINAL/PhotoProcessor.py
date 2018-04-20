import io
import os
import csv

from google.cloud import vision
from google.cloud.vision import types

#We also can read the picture from the post and transform the picture to keywords bu using Google vision API:
#https://cloud.google.com/vision/

#Google vision API will process a picture and return a list of keywords	
#However, the model wo build on CNN does not need the keyword of picture for now. Besides, process the picture take a very long time.
#So, we will keep the picture processing code in our project, but we are not going to use it at this point.
def detect_labels(pic_path):
    client = vision.ImageAnnotatorClient()

    with io.open(pic_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    result=[]
    for label in labels:
        result.append(label.description)
		
    return result

def csv_writer(data, path):
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(data)

##pic_path = 'temp/tempPic.jpg'
##path = "temp/result.csv"
##csv_writer(detect_labels(pic_path),path)