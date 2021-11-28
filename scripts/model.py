""" we can use base-bert, RoBerta, DeBerta as our models encoder architecture"""
from transformers import BertModel, AutoTokenizer, AutoModelWithLMHead, AutoModel
import torch.nn.functional as F
import torch.nn as nn
import torch
import transformers


class BertEncoder(nn.Module):
    def __init__(self, model_type="base-bert"):
        """
        :model_type str: models architecture
        """
        super(BertEncoder, self).__init__()
        if model_type == 'base-bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        elif model_type == 'RoBerta':
            self.bert = AutoModel.from_pretrained('roberta-base')
        elif model_type == "DeBerta":
            self.bert = AutoModel.from_pretrained('microsoft/deberta-base')
        self.feature_size = self.bert.config.hidden_size

    def forward(self, input_ids):
        """

        :param input_ids: List[int], the id of every "token" in the sentence
        :return: the last hidden state, the shape is (batch_size, seq_length, hidden_size)
        """
        last_hidden_state = self.bert(input_ids=input_ids).last_hidden_state
        return last_hidden_state


class OodModel(nn.Module):
    """the models will do aspect based sentiment out of domain"""

    def __init__(self, output_dropout=.1, model_type='base-bert'):
        super(OodModel, self).__init__()
        self.bert = BertEncoder(model_type=model_type)
        self.ffn = nn.Sequential(
            nn.Linear(self.bert.feature_size, self.bert.feature_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.bert.feature_size, 3),
            nn.Softmax()
        )

    def forward(self, batch, device):
        """
        :param batch: tuple of (inputs_ids, labels)
        :param device: GPU or CPU
        :return: loss, num_rows, y_pred, targets
        """
        inputs, targets = batch
        inputs, num_rows = inputs.to(device), inputs.size(0)
        targets = targets.to(device)
        last_hidden_state = self.bert(inputs)
        logits = self.ffn(last_hidden_state)
        logits = logits[:, 0, :]

        # print(logits.shape)
        # print(targets.shape)

        # print(targets)
        # print(logits)

        # loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
        loss = loss_function(logits, targets).cuda()

        y_pred = self.compute_pred(logits)
        targets = targets.max(axis=1)[1]
        return loss, num_rows, y_pred, targets.cpu().numpy()

    @staticmethod
    def compute_pred(logits):

        y_pred = logits.max(axis=1)

        return y_pred[1].cpu().numpy()
