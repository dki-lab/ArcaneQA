import json


'''
Convert the output file from AllenNLP predict command to the proper input for our evaluation script
'''
def process_file_to_json(file_name):
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
    if file_name.__contains__('.txt'):
        json.dump(results, open(file_name.replace('txt', 'json'), 'w'), indent=4)
    else:
        json.dump(results, open(file_name + '.json', 'w'), indent=4)
