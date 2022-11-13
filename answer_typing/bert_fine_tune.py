import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = 1


class BertDataset(Dataset):
    def __init__(self, tokenizer, max_length, data, device=DEVICE):
        super(BertDataset, self).__init__()
        self.tokenizer = tokenizer
        # self.target = self.train_csv.iloc[:, 1]
        self.data = data
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data[index][0]
        y = self.data[index][1]

        inputs = self.tokenizer.encode_plus(
            question,
            y,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        if len(self.data[index]) == 3:
            return {
                'ids': torch.tensor(ids, dtype=torch.long).to(self.device),
                'mask': torch.tensor(mask, dtype=torch.long).to(self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(self.device),
                # 'target': torch.tensor(self.train_csv.iloc[index, 1], dtype=torch.long)
                'target': torch.tensor(self.data[index][2], dtype=torch.long).to(self.device)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long).to(self.device),
                'mask': torch.tensor(mask, dtype=torch.long).to(self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(self.device)
            }


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        out = self.out(o2)

        return out


def finetune(epochs, dataloader, model, loss_fn, optimizer):
    model.train()
    for epoch in range(epochs):
        print(epoch)

        loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for batch, dl in loop:
            ids = dl['ids']
            token_type_ids = dl['token_type_ids']
            mask = dl['mask']
            label = dl['target']
            label = label.unsqueeze(1)

            optimizer.zero_grad()

            output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)

            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()

            pred = torch.where(output >= 0, torch.tensor(1).to(DEVICE), torch.tensor(0).to(DEVICE))

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            num_samples = pred.shape[0]
            accuracy = num_correct / num_samples

            print(
                f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

            # Show progress while training
            loop.set_description(f'Epoch={epoch}/{epochs}')
            loop.set_postfix(loss=loss.item(), acc=accuracy)

        torch.save(model, f"answer_typer_{epoch}.pth")

    return model


def evaluate(model, dataloader):
    model.eval()

    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for batch, dl in loop:
        ids = dl['ids']
        token_type_ids = dl['token_type_ids']
        mask = dl['mask']
        label = dl['target']
        label = label.unsqueeze(1)

        output = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids)
        print(output)
        label = label.type_as(output)

        pred = torch.where(output >= 0, torch.tensor(1).to(DEVICE), torch.tensor(0).to(DEVICE))

        num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
        num_samples = pred.shape[0]
        accuracy = num_correct / num_samples

        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


if __name__ == '__main__':
    training = []
    with open('../data/answer_typing.txt') as f:
        for line in tqdm(f):
            item = []
            line.replace('\n', '')
            fields = line.split('\t')
            # print(fields)
            item.append(fields[0])
            item.append(fields[1])
            item.append(int(fields[2]))
            training.append(item)

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = BertDataset(tokenizer, max_length=100, data=training)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    model = BERT().to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()  # it combines a Sigmoid function and a BCELoss

    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model = finetune(5, dataloader, model, loss_fn, optimizer)

    model = torch.load("answer_typer_4.pth")
    model.to(DEVICE)  # model is loaded on the device it was saved on
