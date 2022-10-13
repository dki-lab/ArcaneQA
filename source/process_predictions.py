import json
import argparse

from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("dataset", help="kbqa dataset: grail_dev, grail_test, graphq, or webq")
parser.add_argument("prediction", help="path to predictions from AllenNLP's predict command")
parser.add_argument("output", help="name of the output file")
args = parser.parse_args()

'''
Convert the output file from AllenNLP predict command to the proper input for our evaluation script
'''
file_name = args.prediction

results = {}
with open(file_name) as f:
    for line in f:
        object = json.loads(line)

        if 'answer' not in object and 'logical_form' not in object:
            continue

        if 'answer' in object:
            results[object['qid']] = {'logical_form': object['logical_form'], 'answer': object['answer']}
        else:
            results[object['qid']] = {'logical_form': object['logical_form'], 'answer': []}

        if 'score' in object:
            results[object['qid']]['score'] = object['score']

if args.dataset == "grail_dev":
    with open("../el_results/grail_dev.json") as f:
        el_results = json.load(f)
    with open("../data/grailqa_v1.0_dev.json") as f:
        data = json.load(f)
    data_map = {}
    for item in data:
        data_map[str(item['qid'])] = item
elif args.dataset == "grail_test":
    with open("../el_results/grail_test.json") as f:
        el_results = json.load(f)
    with open("../data/grailqa_v1.0_test_public.json") as f:
        data = json.load(f)
    data_map = {}
    for item in data:
        data_map[str(item['qid'])] = item
elif args.dataset == "graphq":
    with open("../el_results/graphq_test.json") as f:
        el_results = json.load(f)
    with open("../data/graphquestions_v1_fb15_test_091420.json") as f:
        data = json.load(f)
    data_map = {}
    for item in data:
        data_map[str(item['qid'])] = item
elif args.dataset == "webq":
    with open("../el_results/webq_test.json") as f:
        el_results = json.load(f)
    with open("../data/weqsp_0107.test.json") as f:
        data = json.load(f)
    data_map = {}
    for item in data:
        data_map[str(item['qid'])] = item

grouped_results = defaultdict(lambda: [])
processed_results = {}
for k in results:
    grouped_results[k.split('_')[0]].append(results[k])

print(len(grouped_results))

for group in grouped_results:
    if len(grouped_results[group]) == 1:
        processed_results[group] = grouped_results[group][0]
    else:
        entities = set()
        for m in el_results[group]:
            for e in el_results[group][m]:
                entities.add(e)
        question = data_map[group]['question']

        max_score = -1e32
        entities_in_max = []
        for prediction in grouped_results[group]:
            program = prediction["logical_form"]
            count = 0
            entities_contained = []
            for e in entities:
                if program.__contains__(e):
                    count += 1
                    entities_contained.append(e)

            if count > 1:
                processed_results[group] = prediction
                entities_in_max = entities_contained
                break

            if len(entities_contained) > len(entities_in_max):  # the first priority is to find more entities
                max_score = prediction['score']
                processed_results[group] = prediction
                entities_in_max = entities_contained
            elif len(entities_contained) == len(entities_in_max):
                if prediction['score'] > max_score:
                    max_score = prediction['score']
                    processed_results[group] = prediction
                    entities_in_max = entities_contained

        # replace with better entities; useful only for webqsp
        if args.dataset == "webq":
            if len(entities_in_max) == 1:
                for prediction in grouped_results[group]:
                    entities_contained = []
                    program = prediction["logical_form"]
                    for e in entities:
                        if program.__contains__(e):
                            entities_contained.append(e)
                    if len(entities_contained) == 1:
                        if processed_results[group]['logical_form'].replace(entities_in_max[0],
                                                                            'entity') == program.replace(
                            entities_contained[0], 'entity'):
                            if el_results[group][entities_contained[0]][0] > el_results[group][entities_in_max[0]][0]:
                                print(processed_results[group]['logical_form'])
                                print(prediction['logical_form'])
                                processed_results[group] = prediction
                                entities_in_max = entities_contained

json.dump(processed_results, open(args.output, 'w'))
