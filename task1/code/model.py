import torch
from transformers import BertModel, AutoModel
import torch.nn.functional as F


class TextRCNN_Bert(torch.nn.Module):
    def __init__(self, model_path, label_num):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        # base 为768   large 为1024
        model_dim = 768

        # model_dim = 1024
        self.lstm = torch.nn.LSTM(model_dim, 128, bidirectional=True, batch_first=True)

        self.fc = torch.nn.Linear(128 * 2 + model_dim, label_num)

    def forward(self, idx, attention_mask, token_type_ids):
        outputs = self.bert(idx, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # [batch_size, seq_len, 768]
        hidden_states = outputs.last_hidden_state
        # hidden_states = self.drop(hidden_states)
        out, _ = self.lstm(hidden_states)  # [batch_size, seq_len, 128 * 2]
        out = torch.cat((hidden_states, out), 2)  # [batch_size, seq_len, 128 * 2 + 768]
        out = F.relu(out)  # [batch_size, seq_len, 128 * 2 + 768]
        out = out.permute(0, 2, 1)  # [batch_size, 128 * 2 + 768, seq_len]
        out = F.max_pool1d(out, out.size(2)).squeeze(2)  # [batch_size, 128 * 2 + 768]
        logit = self.fc(out)
        return logit
