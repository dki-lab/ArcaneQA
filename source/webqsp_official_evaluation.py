# process our webqsp predictions for official evaluation
# first download the official eval.py
import json
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--gold", help="path to WebQSP.test.json")
parser.add_argument("--prediction", help="path to predictions from AllenNLP's predict command")
args = parser.parse_args()

legacy_format = []

with open(args.prediction) as f:
    data = json.load(f)

for k in data:
    legacy_format.append({"QuestionId": k, "Answers": data[k]['answer']})

tmp_fn = "tmp_webqsp_official"
json.dump(legacy_format, open(tmp_fn, 'w'))

os.system("python eval.py {} {}".format(args.gold, tmp_fn))
