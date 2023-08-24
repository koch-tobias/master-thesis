# %%
import requests
import time

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

url = "http://10.3.13.137:7070/api/get_relevant_parts/" 

start = time.time()
file_path = config["paths"]["project_path"]
files = {"file": open(file_path, "rb")}
headers = {"accept": "application/json"}

proxies = {
  "http": None,
  "https": None}

response = requests.post(url, files=files, headers=headers, proxies=proxies)
stop = time.time()
training_time = stop - start
print(response.content)