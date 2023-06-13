import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Attention import LinearAttention

import config



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

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)


class ReduceHidden(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(ReduceHidden, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(3 * self.hidden_size, 768)
        self.dropout = nn.Dropout(p = 0.001)

        init_linear_wt(self.linear)

    def forward(self, code_hidden, ast_hidden, name_hidden):
        hidden = torch.cat((code_hidden, ast_hidden, name_hidden), dim=2)
        hidden = self.linear(hidden)
        #hidden = self.dropout(hidden)
        hidden = F.relu(hidden)
        
        return hidden

class MLP(nn.Module):

    def __init__(self, num_i, num_h1, num_h2, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h1)
        self.relu = torch.nn.ReLU()
        #self.relu = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.batch_norm_1 = torch.nn.BatchNorm1d(num_h1)
        self.linear2 = torch.nn.Linear(num_h1, num_o)  # 2个隐层

        init_linear_wt(self.linear1)
        init_linear_wt(self.linear2)
        
        
    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.batch_norm_1(x)
        x = self.linear2(x)
        return x

class Model(nn.Module):

    def __init__(self, code_vocab_size, ast_vocab_size, nl_vocab_size,
                 model_file_path=None, model_state_dict=None, is_eval=False):
        super(Model, self).__init__()

        # vocabulary size for encoders
        self.code_vocab_size = code_vocab_size
        self.ast_vocab_size = ast_vocab_size
        self.is_eval = is_eval

        # init models TF
        self.code_encoder_tf = Encoder(self.code_vocab_size)
        self.name_encoder_tf = Encoder(self.code_vocab_size)
        self.ast_encoder_tf_0 = Encoder(self.ast_vocab_size)
        self.ast_encoder_tf_1 = Encoder(self.ast_vocab_size)
        self.ast_encoder_tf_2 = Encoder(self.ast_vocab_size)
        self.ast_encoder_tf_3 = Encoder(self.ast_vocab_size)
        self.ast_encoder_tf_4 = Encoder(self.ast_vocab_size)
        self.reduce_hidden_tf = ReduceHidden()
            # attnetion & mlp
        self.sent_dim = config.hidden_size
        self.atten_guide_tf = nn.Parameter(torch.Tensor(self.sent_dim).cuda())
        self.atten_guide_tf.data.normal_(0, 1)
        self.atten_tf = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)
        self.MLP_tf = MLP(128 * 3, 128 * 2, None, 128)  # 556, 400, 300, 256


        # init models Torch
        self.code_encoder_torch = Encoder(self.code_vocab_size)
        self.name_encoder_torch = Encoder(self.code_vocab_size)
        self.ast_encoder_torch_0 = Encoder(self.ast_vocab_size)
        self.ast_encoder_torch_1 = Encoder(self.ast_vocab_size)
        self.ast_encoder_torch_2 = Encoder(self.ast_vocab_size)
        self.ast_encoder_torch_3 = Encoder(self.ast_vocab_size)
        self.ast_encoder_torch_4 = Encoder(self.ast_vocab_size)
        self.reduce_hidden_torch = ReduceHidden()

        self.sent_dim = config.hidden_size
        self.atten_guide_torch = nn.Parameter(torch.Tensor(self.sent_dim).cuda())
        self.atten_guide_torch.data.normal_(0, 1)
        self.atten_torch = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)

        self.MLP_torch = MLP(128 * 3 , 128 *2, None, 128)  # 556, 400, 300, 256

        if config.use_cuda:
            device_ids = [1]
            # self.model = self.model.cuda()
            self.code_encoder_tf = torch.nn.DataParallel(self.code_encoder_tf, device_ids=device_ids).cuda()
            self.name_encoder_tf = torch.nn.DataParallel(self.name_encoder_tf, device_ids=device_ids).cuda()
            self.ast_encoder_tf_0 = torch.nn.DataParallel(self.ast_encoder_tf_0, device_ids=device_ids).cuda()
            self.ast_encoder_tf_1 = torch.nn.DataParallel(self.ast_encoder_tf_1, device_ids=device_ids).cuda()
            self.ast_encoder_tf_2 = torch.nn.DataParallel(self.ast_encoder_tf_2, device_ids=device_ids).cuda()
            self.ast_encoder_tf_3 = torch.nn.DataParallel(self.ast_encoder_tf_3, device_ids=device_ids).cuda()
            self.ast_encoder_tf_4 = torch.nn.DataParallel(self.ast_encoder_tf_4, device_ids=device_ids).cuda()
            self.reduce_hidden_tf = torch.nn.DataParallel(self.reduce_hidden_tf, device_ids=device_ids).cuda()
            self.atten_tf = torch.nn.DataParallel(self.atten_tf, device_ids=device_ids).cuda()
            self.MLP_tf = torch.nn.DataParallel(self.MLP_tf, device_ids=device_ids).cuda()

            self.code_encoder_torch = torch.nn.DataParallel(self.code_encoder_torch, device_ids=device_ids).cuda()
            self.name_encoder_torch = torch.nn.DataParallel(self.name_encoder_torch, device_ids=device_ids).cuda()
            self.ast_encoder_torch_0 = torch.nn.DataParallel(self.ast_encoder_torch_0, device_ids=device_ids).cuda()
            self.ast_encoder_torch_1 = torch.nn.DataParallel(self.ast_encoder_torch_1, device_ids=device_ids).cuda()
            self.ast_encoder_torch_2 = torch.nn.DataParallel(self.ast_encoder_torch_2, device_ids=device_ids).cuda()
            self.ast_encoder_torch_3 = torch.nn.DataParallel(self.ast_encoder_torch_3, device_ids=device_ids).cuda()
            self.ast_encoder_torch_4 = torch.nn.DataParallel(self.ast_encoder_torch_4, device_ids=device_ids).cuda()
            self.reduce_hidden_torch = torch.nn.DataParallel(self.reduce_hidden_torch, device_ids=device_ids).cuda()
            self.atten_torch = torch.nn.DataParallel(self.atten_torch, device_ids=device_ids).cuda()
            self.MLP_torch = torch.nn.DataParallel(self.MLP_torch, device_ids=device_ids).cuda()

        if model_file_path:
            state = torch.load(model_file_path)
            self.set_state_dict(state)

        if model_state_dict:
            self.set_state_dict(model_state_dict)

        if is_eval:
            self.code_encoder_tf.eval()
            self.name_encoder_tf.eval()
            self.ast_encoder_tf_0.eval()
            self.ast_encoder_tf_1.eval()
            self.ast_encoder_tf_2.eval()
            self.ast_encoder_tf_3.eval()
            self.ast_encoder_tf_4.eval()
            self.reduce_hidden_tf.eval()
            self.MLP_tf.eval()
            self.atten_tf.eval()

            self.code_encoder_torch.eval()
            self.name_encoder_torch.eval()
            self.ast_encoder_torch_0.eval()
            self.ast_encoder_torch_1.eval()
            self.ast_encoder_torch_2.eval()
            self.ast_encoder_torch_3.eval()
            self.ast_encoder_torch_4.eval()
            self.reduce_hidden_torch.eval()
            self.MLP_torch.eval()
            self.atten_torch.eval()



    def forward(self, batch, batch_size, nl_vocab, is_test=False):
        """

        :param batch:
        :param batch_size:
        :param nl_vocab:
        :param is_test: if True, function will return before decoding
        :return: decoder_outputs: [T, B, nl_vocab_size]
        """
        # batch: [T, B]
        code_batch_tf, code_seq_lens_tf, ast_batch_tf, ast_seq_lens_tf, attention_mask_batch_tf, name_batch_tf, name_seq_lens_tf, index_batch_tf,\
        code_batch_torch, code_seq_lens_torch, ast_batch_torch, ast_seq_lens_torch, attention_mask_batch_torch, name_batch_torch, name_seq_lens_torch, index_batch_torch,\
        code_batch_neg, code_seq_lens_neg, ast_batch_neg, ast_seq_lens_neg, attention_mask_batch_neg, name_batch_neg,\
        nl_batch, nl_seq_lens,  = batch

        attention_mask_batch_tf = torch.tensor(attention_mask_batch_tf, device=config.device).long()
        # encode

        # import ipdb
        # ipdb.set_trace()
        code_outputs_tf, code_hidden_tf = self.code_encoder_tf(code_batch_tf, code_seq_lens_tf)
        name_outputs_tf, name_hidden_tf = self.name_encoder_tf(name_batch_tf, name_seq_lens_tf)
        ast_outputs_0_tf, ast_hidden_0_tf = self.ast_encoder_tf_0(ast_batch_tf[0], ast_seq_lens_tf[0])
        ast_outputs_1_tf, ast_hidden_1_tf = self.ast_encoder_tf_0(ast_batch_tf[1], ast_seq_lens_tf[1])
        ast_outputs_2_tf, ast_hidden_2_tf = self.ast_encoder_tf_0(ast_batch_tf[2], ast_seq_lens_tf[2])
        ast_outputs_3_tf, ast_hidden_3_tf = self.ast_encoder_tf_0(ast_batch_tf[3], ast_seq_lens_tf[3])
        ast_outputs_4_tf, ast_hidden_4_tf = self.ast_encoder_tf_0(ast_batch_tf[4], ast_seq_lens_tf[4])

        ast_outputs_tf = torch.cat((ast_outputs_0_tf, ast_outputs_1_tf, ast_outputs_2_tf, ast_outputs_3_tf, ast_outputs_4_tf), 0)
        # data for decoder
        code_hidden_tf = code_hidden_tf[0] + code_hidden_tf[1]  # [B, H]
        code_hidden_tf = code_hidden_tf.unsqueeze(0)  # [1, B, H]

        name_hidden_tf = name_hidden_tf[0] + name_hidden_tf[1]  # [B, H]
        name_hidden_tf = name_hidden_tf.unsqueeze(0)  # [1, B, H]

        ast_hidden_0_tf = ast_hidden_0_tf[0] + ast_hidden_0_tf[1]  # [B, H]
        ast_hidden_1_tf = ast_hidden_1_tf[0] + ast_hidden_1_tf[1]  # [B, H]
        ast_hidden_2_tf = ast_hidden_2_tf[0] + ast_hidden_2_tf[1]  # [B, H]
        ast_hidden_3_tf = ast_hidden_3_tf[0] + ast_hidden_3_tf[1]  # [B, H]
        ast_hidden_4_tf = ast_hidden_4_tf[0] + ast_hidden_4_tf[1]  # [B, H]


        # ast_hidden_tf = torch.cat((ast_hidden_0_tf, ast_hidden_1_tf, ast_hidden_2_tf, ast_hidden_3_tf, ast_hidden_4_tf),
        #                        dim=1)  # [B, 5 * H]
        # ast_hidden_tf = self.MLP_tf(ast_hidden_tf)
        # ast_hidden_tf = ast_hidden_tf.unsqueeze(0)

        ast_hidden_tf = torch.stack((ast_hidden_0_tf, ast_hidden_1_tf, ast_hidden_2_tf, ast_hidden_3_tf, ast_hidden_4_tf), dim=1) # [B, 5, H]

        atten_guide_tf = torch.unsqueeze(self.atten_guide_torch, dim=1).expand(-1, batch_size)
        atten_guide_tf = atten_guide_tf.transpose(1, 0)
        # import ipdb
        # ipdb.set_trace()
        sent_probs_tf = self.atten_torch(atten_guide_tf, ast_hidden_tf, attention_mask_batch_tf)
        batch_size_tf, srclen_tf, dim_tf = ast_hidden_tf.size()
        sent_probs_tf = sent_probs_tf.view(batch_size_tf, srclen_tf, -1)
        ast_hidden_tf = ast_hidden_tf * sent_probs_tf
        ast_hidden_tf = ast_hidden_tf.sum(dim=1)
        ast_hidden_tf = torch.unsqueeze(ast_hidden_tf, 0)

        # import ipdb
        # ipdb.set_trace()

        decoder_hidden_tf = self.reduce_hidden_tf(code_hidden_tf, ast_hidden_tf, name_hidden_tf)
        # decoder_hidden_tf = self.MLP_tf(decoder_hidden_tf)
        # decoder_hidden_tf = name_hidden_tf


        ################################ TORCH
        ################################
        attention_mask_batch_torch = torch.tensor(attention_mask_batch_torch, device=config.device).long()
        # encode
        # outputs: [T, B, H]
        # hidden: [2, B, H]
        code_outputs_torch, code_hidden_torch = self.code_encoder_torch(code_batch_torch, code_seq_lens_torch)
        name_outputs_torch, name_hidden_torch = self.name_encoder_torch(name_batch_torch, name_seq_lens_torch)
        ast_outputs_0_torch, ast_hidden_0_torch = self.ast_encoder_torch_0(ast_batch_torch[0], ast_seq_lens_torch[0])
        ast_outputs_1_torch, ast_hidden_1_torch = self.ast_encoder_torch_0(ast_batch_torch[1], ast_seq_lens_torch[1])
        ast_outputs_2_torch, ast_hidden_2_torch = self.ast_encoder_torch_0(ast_batch_torch[2], ast_seq_lens_torch[2])
        ast_outputs_3_torch, ast_hidden_3_torch = self.ast_encoder_torch_0(ast_batch_torch[3], ast_seq_lens_torch[3])
        ast_outputs_4_torch, ast_hidden_4_torch = self.ast_encoder_torch_0(ast_batch_torch[4], ast_seq_lens_torch[4])

        ast_outputs_torch = torch.cat(
            (ast_outputs_0_torch, ast_outputs_1_torch, ast_outputs_2_torch, ast_outputs_3_torch, ast_outputs_4_torch), 0)
        # data for decoder
        code_hidden_torch = code_hidden_torch[0] + code_hidden_torch[1]  # [B, H]
        code_hidden_torch = code_hidden_torch.unsqueeze(0)  # [1, B, H]

        name_hidden_torch = name_hidden_torch[0] + name_hidden_torch[1]  # [B, H]
        name_hidden_torch = name_hidden_torch.unsqueeze(0)  # [1, B, H]

        ast_hidden_0_torch = ast_hidden_0_torch[0] + ast_hidden_0_torch[1]  # [B, H]
        ast_hidden_1_torch = ast_hidden_1_torch[0] + ast_hidden_1_torch[1]  # [B, H]
        ast_hidden_2_torch = ast_hidden_2_torch[0] + ast_hidden_2_torch[1]  # [B, H]
        ast_hidden_3_torch = ast_hidden_3_torch[0] + ast_hidden_3_torch[1]  # [B, H]
        ast_hidden_4_torch = ast_hidden_4_torch[0] + ast_hidden_4_torch[1]  # [B, H]


        # ast_hidden_torch = torch.cat((ast_hidden_0_torch, ast_hidden_1_torch, ast_hidden_2_torch, ast_hidden_3_torch, ast_hidden_4_torch),
        #                           dim=1)  # [B, 5 * H]
        # ast_hidden_torch = self.MLP_torch(ast_hidden_torch)
        # ast_hidden_torch = ast_hidden_torch.unsqueeze(0)

        ast_hidden_torch = torch.stack(
            (ast_hidden_0_torch, ast_hidden_1_torch, ast_hidden_2_torch, ast_hidden_3_torch, ast_hidden_4_torch), dim=1)  # [B, 5, H]

        atten_guide_torch = torch.unsqueeze(self.atten_guide_torch, dim=1).expand(-1, batch_size)
        atten_guide_torch = atten_guide_torch.transpose(1, 0)
        # import ipdb
        # ipdb.set_trace()
        sent_probs_torch = self.atten_torch(atten_guide_torch, ast_hidden_torch, attention_mask_batch_torch)
        batch_size_torch, srclen_torch, dim_torch = ast_hidden_torch.size()
        sent_probs_torch = sent_probs_torch.view(batch_size_torch, srclen_torch, -1)
        ast_hidden_torch = ast_hidden_torch * sent_probs_torch
        ast_hidden_torch = ast_hidden_torch.sum(dim=1)
        ast_hidden_torch = torch.unsqueeze(ast_hidden_torch, 0)


        decoder_hidden_torch = self.reduce_hidden_torch(code_hidden_torch, ast_hidden_torch, name_hidden_torch)
        # decoder_hidden_torch = self.MLP_tf(decoder_hidden_torch)
        # decoder_hidden_torch = name_hidden_torch

        ################################ Neg
        ################################

        # encode
        # outputs: [T, B, H]
        # hidden: [2, B, H]
        attention_mask_batch_neg = torch.tensor(attention_mask_batch_neg, device=config.device).long()
        # print(attention_mask_batch_neg.size(), attention_mask_batch_torch.size())
        # print(name_batch_neg.size(), name_batch_torch.size())
        code_outputs_neg, code_hidden_neg = self.code_encoder_torch(code_batch_neg, code_seq_lens_neg)
        ast_outputs_0_neg, ast_hidden_0_neg = self.ast_encoder_torch_0(ast_batch_neg[0], ast_seq_lens_neg[0])
        ast_outputs_1_neg, ast_hidden_1_neg = self.ast_encoder_torch_1(ast_batch_neg[1], ast_seq_lens_neg[1])
        ast_outputs_2_neg, ast_hidden_2_neg = self.ast_encoder_torch_2(ast_batch_neg[2], ast_seq_lens_neg[2])
        ast_outputs_3_neg, ast_hidden_3_neg = self.ast_encoder_torch_3(ast_batch_neg[3], ast_seq_lens_neg[3])
        ast_outputs_4_neg, ast_hidden_4_neg = self.ast_encoder_torch_4(ast_batch_neg[4], ast_seq_lens_neg[4])

        ast_outputs_neg = torch.cat(
            (ast_outputs_0_neg, ast_outputs_1_neg, ast_outputs_2_neg, ast_outputs_3_neg, ast_outputs_4_neg),
            0)
        # data for decoder
        code_hidden_neg = code_hidden_neg[0] + code_hidden_neg[1]  # [B, H]
        code_hidden_neg = code_hidden_neg.unsqueeze(0)  # [1, B, H]

        ast_hidden_0_neg = ast_hidden_0_neg[0] + ast_hidden_0_neg[1]  # [B, H]
        ast_hidden_1_neg = ast_hidden_1_neg[0] + ast_hidden_1_neg[1]  # [B, H]
        ast_hidden_2_neg = ast_hidden_2_neg[0] + ast_hidden_2_neg[1]  # [B, H]
        ast_hidden_3_neg = ast_hidden_3_neg[0] + ast_hidden_3_neg[1]  # [B, H]
        ast_hidden_4_neg = ast_hidden_4_neg[0] + ast_hidden_4_neg[1]  # [B, H]
        # ast_hidden_0 = ast_hidden_0.unsqueeze(0)       # [1, B, H]

        ast_hidden_neg = torch.stack(
            (ast_hidden_0_neg, ast_hidden_1_neg, ast_hidden_2_neg, ast_hidden_3_neg, ast_hidden_4_neg),
            dim=1)  # [B, 5, H]

        atten_guide_neg = torch.unsqueeze(self.atten_guide_torch, dim=1).expand(-1, batch_size)
        atten_guide_neg = atten_guide_neg.transpose(1, 0)
        # import ipdb
        # ipdb.set_trace()
        sent_probs_neg = self.atten_torch(atten_guide_neg, ast_hidden_neg, attention_mask_batch_neg)
        batch_size_neg, srclen_neg, dim_neg = ast_hidden_neg.size()
        sent_probs_neg = sent_probs_neg.view(batch_size_neg, srclen_neg, -1)
        represents_neg = ast_hidden_neg * sent_probs_neg
        represents_neg = represents_neg.sum(dim=1)

        # signatures_hiddens_neg = torch.cat((represents_neg, name_batch_neg), 1)
        # signatures_hiddens_neg = self.MLP_torch(signatures_hiddens_neg)
        signatures_hiddens_neg = represents_neg
        signatures_hiddens_neg = signatures_hiddens_neg.unsqueeze(0)
        # signatures_hiddens_neg = represents_neg.unsqueeze(0)

        decoder_hidden_neg = torch.squeeze(torch.cat((code_hidden_neg, signatures_hiddens_neg), 1), 0)

        return code_outputs_tf, ast_outputs_tf, decoder_hidden_tf, code_hidden_tf, ast_hidden_tf, \
               code_outputs_torch, ast_outputs_torch, decoder_hidden_torch, code_hidden_torch, ast_hidden_torch, \
               code_outputs_neg, ast_outputs_neg, decoder_hidden_neg
        #return code_outputs_tf, ast_outputs_tf, torch.tensor(name_batch_tf, dtype = torch.float64), code_outputs_torch, ast_outputs_torch, torch.tensor(name_batch_torch, dtype = torch.float64)


    def set_state_dict(self, state_dict):

        if not self.is_eval:
            pass
            # ipdb.set_trace()

            self.code_encoder_tf.load_state_dict(state_dict["code_encoder"])
            self.name_encoder_tf.load_state_dict(state_dict["name_encoder"])
            # self.name_encoder_tf.load_state_dict(state_dict["code_encoder"])
            self.ast_encoder_tf_0.load_state_dict(state_dict["ast_encoder_0"])
            self.ast_encoder_tf_1.load_state_dict(state_dict["ast_encoder_1"])
            self.ast_encoder_tf_2.load_state_dict(state_dict["ast_encoder_2"])
            self.ast_encoder_tf_3.load_state_dict(state_dict["ast_encoder_3"])
            self.ast_encoder_tf_4.load_state_dict(state_dict["ast_encoder_4"])
            self.reduce_hidden_tf.load_state_dict(state_dict["reduce_hidden"])
            self.atten_tf.load_state_dict(state_dict['Attention'])
            # self.MLP_tf.load_state_dict(state_dict['MLP'])

            self.code_encoder_torch.load_state_dict(state_dict["code_encoder"])
            self.name_encoder_torch.load_state_dict(state_dict["name_encoder"])
            # self.name_encoder_torch.load_state_dict(state_dict["code_encoder"])
            self.ast_encoder_torch_0.load_state_dict(state_dict["ast_encoder_0"])
            self.ast_encoder_torch_1.load_state_dict(state_dict["ast_encoder_1"])
            self.ast_encoder_torch_2.load_state_dict(state_dict["ast_encoder_2"])
            self.ast_encoder_torch_3.load_state_dict(state_dict["ast_encoder_3"])
            self.ast_encoder_torch_4.load_state_dict(state_dict["ast_encoder_4"])
            self.reduce_hidden_torch.load_state_dict(state_dict["reduce_hidden"])
            self.atten_torch.load_state_dict(state_dict['Attention'])
            # self.MLP_torch.load_state_dict(state_dict['MLP'])
        else:
            self.code_encoder_tf.load_state_dict(state_dict["code_encoder_tf"])
            self.name_encoder_tf.load_state_dict(state_dict["name_encoder_tf"])
            self.ast_encoder_tf_0.load_state_dict(state_dict["ast_encoder_tf_0"])
            self.ast_encoder_tf_1.load_state_dict(state_dict["ast_encoder_tf_1"])
            self.ast_encoder_tf_2.load_state_dict(state_dict["ast_encoder_tf_2"])
            self.ast_encoder_tf_3.load_state_dict(state_dict["ast_encoder_tf_3"])
            self.ast_encoder_tf_4.load_state_dict(state_dict["ast_encoder_tf_4"])
            self.reduce_hidden_tf.load_state_dict(state_dict["reduce_hidden_tf"])
            # self.atten_tf.load_state_dict(state_dict['attention_tf'])
            self.MLP_tf.load_state_dict(state_dict['MLP_tf'])

            self.code_encoder_torch.load_state_dict(state_dict["code_encoder_torch"])
            self.name_encoder_torch.load_state_dict(state_dict["name_encoder_torch"])
            self.ast_encoder_torch_0.load_state_dict(state_dict["ast_encoder_torch_0"])
            self.ast_encoder_torch_1.load_state_dict(state_dict["ast_encoder_torch_1"])
            self.ast_encoder_torch_2.load_state_dict(state_dict["ast_encoder_torch_2"])
            self.ast_encoder_torch_3.load_state_dict(state_dict["ast_encoder_torch_3"])
            self.ast_encoder_torch_4.load_state_dict(state_dict["ast_encoder_torch_4"])
            self.reduce_hidden_torch.load_state_dict(state_dict["reduce_hidden_torch"])
            # self.atten_torch.load_state_dict(state_dict['attention_torch'])
            self.MLP_torch.load_state_dict(state_dict['MLP_torch'])
