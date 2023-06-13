from googletrans import Translator
import time
import random
from pandarallel import pandarallel
from tqdm import tqdm
import json
import pandas as pd
from utils import load_file, save_data

pandarallel.initialize(nb_workers=1, progress_bar=True)
translator = Translator(service_urls=['translate.google.com'])

def translate_question(question):
    while True:
        try:
            question = translator.translate(question, src='en', dest='vi').text
        except:
            continue
        break

    return question


def translate_context(samples):
    translated_samples = []
    for sample in samples:
        try:
            if random.randint(0, 100) > 50:
                sample["passage_text"] = translator.translate(sample["passage_text"], src='en', dest='vi').text
            translated_samples.append(sample)
        except:
            time.sleep(0.5)
            continue
    
    return translated_samples

path = "data/ir/train/ms-macro-train_v1.1.json"
data = load_file(path)

translated_data = []
for index, sample in enumerate(tqdm(data[10000:])):
    try:
        sample["query"] = translate_question(sample["query"])
        sample["passages"] = translate_context(sample["passages"])
        
        translated_data.append(sample)
        
        if (index+1) % 100 == 0:
            _path = f'/home/tuyendv/Desktop/reranker/data/ir/{path.split("/")[-1].replace(".json", "-translated.json")}'
            save_data(_path, data=translated_data)
    except:
        time.sleep(1)
        continue