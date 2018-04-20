import requests
import time
import json
import random
import csv
import re
import string
import time
#import PhotoProcessor as pp
#import PhotoDownloader as pd
import os

#create data by using facebook graph API
#https://developers.facebook.com/docs/graph-api

#You will have to use a API token to access to facebook graph API. Read this for how to get token:
#https://towardsdatascience.com/how-to-use-facebook-graph-api-and-extract-data-using-python-1839e19d6999

token = 'EAACEdEose0cBANe5qbZAyxkvjBjYoOBxUVLMwZA1hcjIckAI0yieNfjyweBg41UE2AGocx6mAZAywuZBZCzZCF2ggr1YIK3gl2uNg6B8BZC7m5AK7trOdQHk7WGFgHRZB7DJThfF6jXPtYQXjKAXPY4EvQMKHk7Uy7Q7stsdUB48CWdh2W0FCK8CZAFJ9UYmcfCcZD'

datasize = 4000
pic_path = 'temp/tempPic.jpg'


def reqest_facebook(req):
    url = 'https://graph.facebook.com/v2.12/' + req + '?fields=posts.limit(1){picture,message,created_time}&access_token=' + token
    r = requests.get(url)
    return r

	
#This following code will go through all posts of an account and return the text with a label	
def shot_target(target):
    result = reqest_facebook(target).json()

    translation = str.maketrans("","", string.punctuation)
    try:
	    #first post
        message = result['posts']['data'][0]['message']
        message = clean_str(message)
        message = message.translate(translation)
        time = result['posts']['data'][0]['created_time']
        pictureLink = result['posts']['data'][0]['picture']

#        print("Downloading image"+str(index))
#        pd.download_picture(pictureLink,pic_path)
#        print("Analysing image"+str(index))
#        picInfo=pp.detect_labels(pic_path)
#        print("Deleting image"+str(index))
#        os.remove(pic_path)
        print(len(data))
        data.append([message,target])
    except KeyError as e:
        if e.args[0] == "next":
            print("reach the end")
        else:
            print("connect error, skip this post")
		
    #second post
    try:
        r=requests.get(result['posts']['paging']['next'])
        result = r.json()
        new_messg = result['data'][0]['message']
        new_messg = clean_str(new_messg)
        new_messg = new_messg.translate(translation)
        new_time = result['data'][0]['created_time']
        new_picture = result['data'][0]['picture']

#        print("Downloading image"+str(index))
#        pd.download_picture(new_picture,pic_path)
#        print("Analysing image"+str(index))
#        picInfo=pp.detect_labels(pic_path)
#        print("Deleting image"+str(index))
#        os.remove(pic_path)
        print(len(data))   
        data.append([new_messg,target])
    except KeyError as e:
        if e.args[0] == "next":
            print("reach the end")
        else:
            print("connect error, skip this post")
            
		
    ##following post
    size = len(data)
    while True:
        try:
            r = requests.get(result['paging']['next'])
            result = r.json()
            new_messg = result['data'][0]['message']
            new_messg = clean_str(new_messg)
            new_messg = new_messg.translate(translation)
            new_time = result['data'][0]['created_time']
            new_picture = result['data'][0]['picture']

#            print("Downloading image"+str(index))
#            pd.download_picture(new_picture,pic_path)
#            print("Analysing image"+str(index))
#            picInfo=pp.detect_labels(pic_path)
#            print("Deleting image"+str(index))
#            os.remove(pic_path)
            print(len(data))
            data.append([new_messg,target])
            if len(data) > size+datasize:
                print("reach the limit")
                break
        except KeyError as e:
            if e.args[0] == "next":
                print("reach the end")
                break
            else:
                print("connect error, skip this post")
            continue

			
#pre-process text
def clean_str(string):
    """
    from: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()			
			
def main(target__list,result_path):
    for target in target__list:
            print(target)
            shot_target(target)
			
    myFile = open(result_path, 'w', newline='', encoding='utf-8')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(['Text','label'])
        for row in data:
            writer.writerow(row)
    print("Writing complete")



data = []
target__list= ['ellentv','linkedin']
result_path = 'temp/FacebookData3.csv'
main(target__list,result_path)

data = []
target__list= ['chanel','warcraft']
result_path = 'temp/FacebookData1.csv'
main(target__list,result_path)
	
data = []
target__list= ['leagueoflegends','dota2']
result_path = 'temp/FacebookData2.csv'
main(target__list,result_path)

data = []
target__list= ['facebook','google']
result_path = 'temp/FacebookData4.csv'
main(target__list,result_path)

data = []
target__list= ['steam','microsoft']
result_path = 'temp/FacebookData5.csv'
main(target__list,result_path)
