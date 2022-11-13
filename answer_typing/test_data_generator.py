import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import random

path = str(Path(__file__).parent.absolute())


with open(path + '/../vocab_files/grailqa.json') as f:
    schema_items = json.load(f)
    classes = schema_items["classes"]
    # set(data["relations"]), set(data["classes"]), set(data["attributes"])
    print(len(classes))
    classes = list(classes)

with open("../data/grailqa_v1.0_dev.json") as f:
    data = json.load(f)

def form(question, y):
    return question + '\t' + y

test_data = []

for item in data:
    question = item['question'].replace('\n', '')
    for c in classes:
        test_data.append(form(question, c))

# with open("../data/answer_typing_test.txt", 'w') as f:
with open("../data/answer_typing_dev.txt", 'w') as f:
    for item in tqdm(test_data):
        f.write(item)
        f.write('\n')
    f.close()

