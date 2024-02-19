import json
from transformers import BertTokenizer, AdamW, optimization, AutoTokenizer
import numpy as np
from others.semeval.task1.code.model import TextRCNN_Bert
import torch
from tqdm import tqdm


def read_erc(input_path):
    k = -2
    """

    :param input_path:
    :return:
    """
    data_list = []
    with open(input_path, encoding="utf-8") as file_read:
        data_dict = json.load(file_read)
        for line in data_dict:
            # 自己说的话
            tmp_dict = {}
            n1 = np.array(line['speakers'])
            n2 = np.nan_to_num(n1, nan="someone")
            # 多人对话的内容
            # 这里要控制，多少个人来进行
            tmp_begin = "The following is 's conversation history."
            tmp_list = []
            for index_i, name_i, text_i in zip(range(len(line['utterances'])), n2, line['utterances']):
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

        file_read.close()
    return data_list


if __name__ == '__main__':

    label_name = ['disgust', 'contempt', 'anger', 'neutral', 'joy', 'sadness', 'fear', 'surprise']

    train_path = "../data/MaSaC_test_erc.json"
    train_list = read_erc(train_path)
    label_num = 8
    MAX_LEN = 512
    device = "cuda"
    model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
    save_dir = "./best_stage_new.pth"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = TextRCNN_Bert(model_path, label_num)

    model.load_state_dict(torch.load(save_dir))
    model.to(device)
    label_list = []
    for line in tqdm(train_list):
        inputs = tokenizer.encode_plus(line, None, add_special_tokens=True, max_length=MAX_LEN,
                                       pad_to_max_length=True,
                                       return_token_type_ids=True)

        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        label_list.extend(outputs.cpu().detach().numpy().argmax(1).tolist())

    with open("answer1.txt", "w", encoding="utf-8") as file_write:
        for label_i in label_list:
            file_write.write(label_name[label_i] + "\n")
    file_write.close()
