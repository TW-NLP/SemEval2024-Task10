import torch
from torch.utils.data import Dataset


class CLSDataset(Dataset):
    def __init__(self, data_list, label_list, tokenizer, max_len):
        """

        :param data_list:
        :param label_list:
        :param tokenizer:
        :param max_len:
        """
        self.data_list = data_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        data_item = self.data_list[item]
        label_item = self.label_list[item]
        inputs = self.tokenizer.encode_plus(data_item, None, add_special_tokens=True, max_length=self.max_len,
                                            pad_to_max_length=True,
                                            return_token_type_ids=True)
        return {
            "input_ids": torch.tensor(inputs['input_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(inputs['attention_mask'], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            "labels": torch.tensor(label_item)
        }

    def __len__(self):
        return len(self.data_list)
