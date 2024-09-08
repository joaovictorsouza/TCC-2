import torch
import torch.nn as nn
from hparams import hp
from transformers import AutoModel

class Net(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attention_mask=None):
        '''
        x: (N, T). int64
        attention_mask: (N, T). int64

        Returns
        logits: (N, n_classes)
        y_hat: (N, n_candidates)
        y_hat_prob: (N, n_candidates)
        '''
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output  # (N, hidden_size)
        
        logits = self.classifier(pooler_output)  # (N, n_classes)
        activated = self.softmax(logits)  # (N, n_classes)

        y_hat_prob, y_hat = activated.sort(-1, descending=True)
        y_hat_prob = y_hat_prob[:, :hp.n_candidates]
        y_hat = y_hat[:, :hp.n_candidates]

        return logits, y_hat, y_hat_prob