#!/usr/bin/python3
import requests
import time
import os

url = "http://localhost:9696/predict"

#image_url = { "url": "https://upload.wikimedia.org/wikipedia/commons/5/58/Santoku_knife.jpg" }
cwd = os.getcwd()
full_url = "file://" + cwd + "/data/images/4138.jpg"
image_url = { "url": full_url }
#image_url = { "url": "file:///home/dietmar/Documents/ZoomCamp/capstone1/data/images/4138.jpg" }

while True:
    time.sleep(0.1)
    response = requests.post(url, json=image_url).json()
    print(response)
