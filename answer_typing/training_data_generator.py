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

with open("../data/grailqa_v1.0_train.json") as f:
    data = json.load(f)

with open('../ontology/domain_dict', 'r') as f:
    domain_dict = json.load(f)
with open('../ontology/domain_info', 'r') as f:
    constants_to_domain = defaultdict(lambda: None)
    constants_to_domain.update(json.load(f))


def form(question, y, label):
    return question + '\t' + y + '\t' + str(label)


training_data = []

for item in data:
    question = item['question'].replace('\n', '')
    negatives = set()
    for node in item['graph_query']['nodes']:
        if node['question_node'] == 1:
            positive = node['class']
            domain = constants_to_domain[positive]
            domain_neg = set(domain_dict[domain]).intersection(classes)
            if len(domain_neg) > 5:
                negatives.update(random.sample(domain_neg, 5))
            else:
                negatives.update(domain_neg)
            negatives.update(random.sample(classes, 5))
            if positive in negatives:
                negatives.remove(positive)

            break

    training_data.append(form(question, positive, 1))
    for c in negatives:
        training_data.append(form(question, c, 0))


with open("../data/answer_typing.txt", 'w') as f:
    for item in tqdm(training_data):
        f.write(item)
        f.write('\n')
    f.close()

