import json
import torch
import transformers
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from answer_typing.bert_fine_tune import BertDataset, BERT

path = str(Path(__file__).parent.absolute())
DEVICE = 1

with open(path + '/../vocab_files/grailqa.json') as f:
    schema_items = json.load(f)
    classes = schema_items["classes"]
    print(len(classes))
    classes = list(classes)


tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


# def process(question, batch_size=32):  # inefficient... plz use PyTorch's dataloader...
#     ids_list = []
#     token_type_ids_list = []
#     mask_list = []
#     for i, c in enumerate(classes):
#         inputs = tokenizer.encode_plus(
#             question,
#             # "travel.hotel_brand",
#             c,
#             pad_to_max_length=True,
#             add_special_tokens=True,
#             return_attention_mask=True,
#             max_length=100,
#             truncation=True
#         )
#         ids_list.append(torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE))
#         token_type_ids_list.append(torch.tensor(inputs["token_type_ids"], dtype=torch.long).to(DEVICE))
#         mask_list.append(torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE))
#
#         if (i + 1) % batch_size == 0 or i == (len(classes) - 1):
#             ids = torch.stack(ids_list)
#             token_type_ids = torch.stack(token_type_ids_list)
#             mask = torch.stack(mask_list)
#
#             ids_list = []
#             token_type_ids_list = []
#             mask_list = []
#
#             yield {
#                 'ids': torch.tensor(ids, dtype=torch.long).to(DEVICE),
#                 'mask': torch.tensor(mask, dtype=torch.long).to(DEVICE),
#                 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(DEVICE)
#             }


test = []
# with open('../data/answer_typing_test_gq1.txt') as f:
with open('../data/answer_typing_test.txt') as f:
    for line in tqdm(f):
        item = []
        line.replace('\n', '')
        fields = line.split('\t')
        # print(fields)
        item.append(fields[0])
        item.append(fields[1])
        test.append(item)

count = 0
model = torch.load("answer_typer_4.pth")
model.eval()
model.to(DEVICE)

test_dataset = BertDataset(tokenizer, max_length=100, data=test)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32)

loop = tqdm(enumerate(test_dataloader), leave=False, total=len(test_dataloader))
predictions = []
for batch, dl in loop:
    # if batch > 50:
    #     break
    ids = dl['ids']
    token_type_ids = dl['token_type_ids']
    mask = dl['mask']

    output = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids)

    for o in output:
        predictions.append(float(o))

with open("predictions.test.txt", 'w') as f:
# with open("predictions.dev.txt", 'w') as f:
    for p in predictions:
        f.write(str(p))
        f.write('\n')
    f.close()
