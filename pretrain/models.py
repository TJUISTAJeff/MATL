import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import random
from Attention import LinearAttention
import config
import utils


def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):

    linear.weight.data.normal_(std=config.init_normal_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.init_normal_std)


def init_wt_normal(wt):

    wt.data.normal_(std=config.init_normal_std)


def init_wt_uniform(wt):

    wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)


class Encoder(nn.Module):


    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        # vocab_size: config.code_vocab_size for code encoder, size of sbt vocabulary for ast encoder
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        embedded = self.embedding(inputs)   # [T, B, embedding_dim]
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # outputs: [T, B, H]
        # hidden: [2, B, H]
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)


class ReduceHidden(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(ReduceHidden, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(3 * self.hidden_size, 256)

        init_linear_wt(self.linear)

    def forward(self, code_hidden, ast_hidden, name_hidden):

        hidden = torch.cat((code_hidden, ast_hidden, name_hidden), dim=2)
        hidden = self.linear(hidden)
        hidden = F.relu(hidden)
        return hidden


class MLP(nn.Module):

    def __init__(self, num_i, num_h1, num_h2, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h1)
        self.relu = torch.nn.ReLU()
        # self.relu = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.batch_norm_1 = torch.nn.BatchNorm1d(num_h1)
        self.linear2 = torch.nn.Linear(num_h1, num_o)  # 2个隐层

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.batch_norm_1(x)
        x = self.linear2(x)
        return x

class Attention(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True)   # [H]
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):

        time_step, batch_size, _ = encoder_outputs.size()
        h = hidden.repeat(time_step, 1, 1).transpose(0, 1)  # [B, T, H]
        encoder_outputs = encoder_outputs.transpose(0, 1)   # [B, T, H]
        attn_energies = self.score(h, encoder_outputs)      # [B, T]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        """
        calculate the attention scores of each word
        :param hidden: [B, T, H]
        :param encoder_outputs: [B, T, H]
        :return: energy: scores of each word in a batch, [B, T]
        """
        # after cat: [B, T, 2*H]
        # after attn: [B, T, H]
        # energy: [B, T, H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2)     # [B, H, T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)      # [B, 1, H]
        energy = torch.bmm(v, energy)   # [B, 1, T]
        return energy.squeeze(1)



class Model(nn.Module):

    def __init__(self, code_vocab_size, ast_vocab_size, nl_vocab_size,
                 model_file_path=None, model_state_dict=None, is_eval=False):
        super(Model, self).__init__()

        # vocabulary size for encoders
        self.code_vocab_size = code_vocab_size
        self.ast_vocab_size = ast_vocab_size
        self.is_eval = is_eval

        # init models
        self.code_encoder = Encoder(self.code_vocab_size)
        self.ast_encoder_0 = Encoder(self.ast_vocab_size)
        self.ast_encoder_1 = Encoder(self.ast_vocab_size)
        self.ast_encoder_2 = Encoder(self.ast_vocab_size)
        self.ast_encoder_3 = Encoder(self.ast_vocab_size)
        self.ast_encoder_4 = Encoder(self.ast_vocab_size)
        self.name_encoder = Encoder(self.code_vocab_size)
        self.MLP = MLP(3 * config.hidden_size, 2 * config.hidden_size, None, config.hidden_size)  # 556, 400, 300, 256
        self.reduce_hidden = ReduceHidden()

        self.sent_dim = config.hidden_size
        self.atten_guide = nn.Parameter(torch.Tensor(self.sent_dim).cuda())
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)
        # attnetion & mlp

        if config.use_cuda:
            device_ids = [1]

            self.code_encoder = torch.nn.DataParallel(self.code_encoder, device_ids=device_ids).cuda()
            self.name_encoder = torch.nn.DataParallel(self.name_encoder, device_ids=device_ids).cuda()
            self.ast_encoder_0 = torch.nn.DataParallel(self.ast_encoder_0, device_ids=device_ids).cuda()
            self.ast_encoder_1 = torch.nn.DataParallel(self.ast_encoder_1, device_ids=device_ids).cuda()
            self.ast_encoder_2 = torch.nn.DataParallel(self.ast_encoder_2, device_ids=device_ids).cuda()
            self.ast_encoder_3 = torch.nn.DataParallel(self.ast_encoder_3, device_ids=device_ids).cuda()
            self.ast_encoder_4 = torch.nn.DataParallel(self.ast_encoder_4, device_ids=device_ids).cuda()
            self.atten = torch.nn.DataParallel(self.atten, device_ids=device_ids).cuda()
            self.reduce_hidden = torch.nn.DataParallel(self.reduce_hidden, device_ids=device_ids).cuda()
            self.MLP = torch.nn.DataParallel(self.MLP, device_ids=device_ids).cuda()


        if model_file_path:
            state = torch.load(model_file_path)
            self.set_state_dict(state)

        if model_state_dict:
            self.set_state_dict(model_state_dict)

        if is_eval:
            self.code_encoder.eval()
            self.name_encoder.eval()
            self.ast_encoder_0.eval()
            self.ast_encoder_1.eval()
            self.ast_encoder_2.eval()
            self.ast_encoder_3.eval()
            self.ast_encoder_4.eval()
            self.reduce_hidden.eval()
            self.MLP.eval()



    def forward(self, batch, batch_size, nl_vocab, is_test=False):
        """

        :param batch:
        :param batch_size:
        :param nl_vocab:
        :param is_test: if True, function will return before decoding
        :return: decoder_outputs: [T, B, nl_vocab_size]
        """
        # batch: [T, B]
        code_batch, code_seq_lens, ast_batch, ast_seq_lens, attention_mask_batch, nl_batch, nl_seq_lens, name_batch, name_seq_lens, result_batch = batch

        attention_mask_batch = torch.tensor(attention_mask_batch, device=config.device).long()

        # encode
        # outputs: [T, B, H]
        # hidden: [2, B, H]
        code_outputs, code_hidden = self.code_encoder(code_batch, code_seq_lens)
        name_outputs, name_hidden = self.name_encoder(name_batch, name_seq_lens)
        ast_outputs_0, ast_hidden_0 = self.ast_encoder_0(ast_batch[0], ast_seq_lens[0])
        ast_outputs_1, ast_hidden_1 = self.ast_encoder_0(ast_batch[1], ast_seq_lens[1])
        ast_outputs_2, ast_hidden_2 = self.ast_encoder_0(ast_batch[2], ast_seq_lens[2])
        ast_outputs_3, ast_hidden_3 = self.ast_encoder_0(ast_batch[3], ast_seq_lens[3])
        ast_outputs_4, ast_hidden_4 = self.ast_encoder_0(ast_batch[4], ast_seq_lens[4])

        ast_outputs = torch.cat(
            (ast_outputs_0, ast_outputs_1, ast_outputs_2, ast_outputs_3, ast_outputs_4), 0)
        # data for decoder
        code_hidden = code_hidden[0] + code_hidden[1]   # [B, H]
        code_hidden = code_hidden.unsqueeze(0)          # [1, B, H]

        name_hidden = name_hidden[0] + name_hidden[1]  # [B, H]
        name_hidden = name_hidden.unsqueeze(0)  # [1, B, H]

        ast_hidden_0 = ast_hidden_0[0] + ast_hidden_0[1]  # [B, H]
        ast_hidden_1 = ast_hidden_1[0] + ast_hidden_1[1]  # [B, H]
        ast_hidden_2 = ast_hidden_2[0] + ast_hidden_2[1]  # [B, H]
        ast_hidden_3 = ast_hidden_3[0] + ast_hidden_3[1]  # [B, H]
        ast_hidden_4 = ast_hidden_4[0] + ast_hidden_4[1]  # [B, H]

        ast_hidden = torch.stack(
            (ast_hidden_0, ast_hidden_1, ast_hidden_2, ast_hidden_3, ast_hidden_4), dim=1)  # [B, 5, H]
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        # import ipdb
        # ipdb.set_trace()
        sent_probs = self.atten(atten_guide, ast_hidden, attention_mask_batch)
        batch_size, srclen, dim = ast_hidden.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        ast_hidden = ast_hidden * sent_probs
        ast_hidden = ast_hidden.sum(dim=1)
        ast_hidden = torch.unsqueeze(ast_hidden, 0)

        # ast_hidden = torch.cat((ast_hidden_0, ast_hidden_1, ast_hidden_2, ast_hidden_3, ast_hidden_4), dim=1) # [B, 5 * H]
        # ast_hidden = self.MLP(ast_hidden)
        # ast_hidden = ast_hidden.unsqueeze(0)

        # import ipdb
        # ipdb.set_trace()
        decoder_hidden = self.reduce_hidden(code_hidden, ast_hidden, name_hidden)  # [1, B, H]


        return code_outputs, ast_outputs, decoder_hidden


    def set_state_dict(self, state_dict):
        self.code_encoder.load_state_dict(state_dict["code_encoder"])
        self.name_encoder.load_state_dict(state_dict["name_encoder"])
        self.ast_encoder_0.load_state_dict(state_dict["ast_encoder_0"])
        self.ast_encoder_1.load_state_dict(state_dict["ast_encoder_1"])
        self.ast_encoder_2.load_state_dict(state_dict["ast_encoder_2"])
        self.ast_encoder_3.load_state_dict(state_dict["ast_encoder_3"])
        self.ast_encoder_4.load_state_dict(state_dict["ast_encoder_4"])
        self.reduce_hidden.load_state_dict(state_dict["reduce_hidden"])
        self.atten.load_state_dict(state_dict["Attention"])
        self.MLP.load_state_dict(state_dict["MLP"])

