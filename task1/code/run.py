import json
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, optimization, AutoTokenizer
import numpy as np
from task1.code.data_loader import CLSDataset
from task1.code.model import TextRCNN_Bert
from task1.code.test import test
from task1.code.train import train
import torch


def read_erc(input_path):
    """

    :param input_path:
    :return:
    """
    # 这里要控制，多少个人来进行
    k = -2
    data_list = []
    label_list = []
    label_dict = {}
    with open(input_path, encoding="utf-8") as file_read:
        data_dict = json.load(file_read)
        for line in data_dict:
            # 自己说的话
            tmp_dict = {}
            n1 = np.array(line['speakers'])
            n2 = np.nan_to_num(n1, nan="someone")
            tmp_begin = "The following is 's conversation history."
            tmp_list = []
            for index_i, name_i, text_i, label_i in zip(range(len(line['utterances'])), n2, line['utterances'],
                                                        line['emotions']):
                tmp_list.append(name_i + ":" + text_i)
                tmp_str = name_i + ":" + text_i
                if name_i not in tmp_dict:
                    tmp_dict[name_i] = []
                    tmp_dict[name_i].append(tmp_str)
                    data_list.append(tmp_begin + '.'.join(
                        tmp_dict[name_i][k:]))
                else:
                    tmp_dict[name_i].append(tmp_str)
                    data_list.append(tmp_begin + '.'.join(
                        tmp_dict[name_i][k:]))

                if label_i not in label_dict:
                    label_dict[label_i] = 0
                else:
                    label_dict[label_i] += 1
                label_list.append(label_name.index(label_i))

        file_read.close()
        print(label_dict)
    return data_list, label_list


if __name__ == '__main__':

    label_name = ['disgust', 'contempt', 'anger', 'neutral', 'joy', 'sadness', 'fear', 'surprise']

    train_path = "../data/MaSaC_train_erc.json"
    val_path = "../data/MaSaC_val_erc.json"

    train_list, train_label = read_erc(train_path)
    val_list, val_label = read_erc(val_path)

    # 参数的设置
    MAX_LEN = 512
    train_batch_size = 16
    dev_batch_size = 16
    LEARNING_RATE = 3e-5

    label_num = len(label_name)
    device = "cuda"
    EPOCHS = 10

    max_cro = 0
    # model url(https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
    model_path = "nlptown/bert-base-multilingual-uncased-sentiment"

    save_dir = "./best_stage_new.pth"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = TextRCNN_Bert(model_path, label_num)

    model.to(device)
    train_dataset = CLSDataset(train_list, train_label, tokenizer, MAX_LEN)
    dev_dataset = CLSDataset(val_list, val_label, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dev_dataset, batch_size=dev_batch_size, shuffle=False)

    #
    loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optimization.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=int(
                                                                 0.05 * EPOCHS * len(train_loader)),
                                                             num_training_steps=EPOCHS * len(train_loader))

    for epoch in range(EPOCHS):
        train(train_loader, model, loss, optimizer, scheduler, device)
        f1_macro = test(test_loader, model, device)
        print(f1_macro)
        if f1_macro > max_cro:
            max_cro = f1_macro
            torch.save(model.state_dict(), save_dir)
