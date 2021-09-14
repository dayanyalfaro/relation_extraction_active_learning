import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, config):
        """
        type_rnn: RNN, GRU, LSTM
        """
        super(RNN, self).__init__()

        self.input_size = config.model.input_size
        self.hidden_size = config.model.hidden_size // 2 if config.model.bidirectional else config.model.hidden_size
        self.num_layers = config.model.num_layers
        self.dropout = config.model.dropout
        self.bidirectional = config.model.bidirectional
        self.last_layer_hn = config.model.last_layer_hn
        self.type_rnn = config.model.type_rnn

        rnn = eval(f'nn.{self.type_rnn}')
        self.rnn = rnn(input_size=self.input_size,
                       hidden_size=self.hidden_size,
                       num_layers=self.num_layers,
                       dropout=self.dropout,
                       bidirectional=self.bidirectional,
                       bias=True,
                       batch_first=True)

    def forward(self, x, x_len):
        """
        :param x: torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H_in] Generally the value after embedding
        :param x_len: torch.Tensor [L] Sentence length value that has been sorted
        :return:
        output: torch.Tensor [B, L, H_out] The result of using sequence annotation
        hn:     torch.Tensor [B, N, H_out] / [B, H_out] The result of classification, when last_layer_hn is the result of the last layer
        """
        B, L, _ = x.size()
        H, N = self.hidden_size, self.num_layers

        x = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=True)
        output, hn = self.rnn(x)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=L)

        if self.type_rnn == 'LSTM':
            hn = hn[0]
        if self.bidirectional:
            hn = hn.view(N, 2, B, H).transpose(1, 2).contiguous().view(N, B, 2 * H).transpose(0, 1)
        else:
            hn = hn.transpose(0, 1)
        if self.last_layer_hn:
            hn = hn[:, -1, :]

        return output, hn
