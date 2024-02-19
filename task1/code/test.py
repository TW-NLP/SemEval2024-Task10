import torch
from sklearn import metrics
from tqdm import tqdm
import numpy as np


def test(test_dataloader, model, device):
    """

    :param test_dataloader:
    :param model:
    :param device:
    :return:
    """

    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            fin_outputs.extend(outputs.cpu().detach().numpy().argmax(1).tolist())
            fin_targets.extend(labels.cpu().detach().numpy().tolist())
    acc = metrics.accuracy_score(fin_targets, fin_outputs)

    f1_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')
    print("acc:", acc)
    return f1_macro
