import time
import re
import torch
import itertools
import os
import pickle
import numpy as np
import nltk
import config


# special vocabulary symbols

_PAD = '<PAD>'
_SOS = '<s>'    # start of sentence
_EOS = '</s>'   # end of sentence
_UNK = '<UNK>'  # OOV word

_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]


class Vocab(object):

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0
        self.add_sentence(_START_VOCAB)     # add special symbols

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, max_vocab_size=None):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        # trim according to minimum count of words
        if config.trim_vocab_min_count:
            keep_words += _START_VOCAB
            # filter words
            for word, count in self.word2count.items():
                if count >= config.vocab_min_count:
                    keep_words.append(word)

        # trim according to maximum size of vocabulary
        if config.trim_vocab_max_size:
            if max_vocab_size is None:
                raise Exception('Parameter \'max_vocab_size\'must be passed if \'config.trim_vocab_max_size\' is True')
            if self.num_words <= max_vocab_size:
                return
            for special_symbol in _START_VOCAB:
                self.word2count.pop(special_symbol)
            keep_words = list(self.word2count.items())
            keep_words = sorted(keep_words, key=lambda item: item[1], reverse=True)
            keep_words = keep_words[: max_vocab_size - len(_START_VOCAB)]
            keep_words = _START_VOCAB + [word for word, _ in keep_words]

        # reinitialize
        self.word2index.clear()
        self.word2count.clear()
        self.index2word.clear()
        self.num_words = 0
        self.add_sentence(keep_words)

    def save(self, name):

        path = os.path.join(config.vocab_dir, name)
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def save_txt(self, name):
        txt_path = os.path.join(config.vocab_dir, name)
        with open(txt_path, 'w', encoding='utf-8') as file:
            for word, _ in self.word2index.items():
                file.write(word + '\n')

    def __len__(self):
        return self.num_words


class EarlyStopping(object):

    def __init__(self, patience=config.early_stopping_patience, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.min_valid_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, valid_loss):

        if self.min_valid_loss is None:
            self.min_valid_loss = valid_loss
        elif valid_loss > self.min_valid_loss - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}.\n'.format(self.counter, self.patience))
            config.logger.info('EarlyStopping counter: {} out of {}.'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print('Early stop.\n')
                config.logger.info('Early stop.')
        else:
            self.min_valid_loss = valid_loss
            self.counter = 0


def load_vocab_pk(file_name) -> Vocab:

    path = os.path.join(config.vocab_dir, file_name)
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    if not isinstance(vocab, Vocab):
        raise Exception('Pickle file: \'{}\' is not an instance of class \'Vocab\''.format(path))
    return vocab


def get_timestamp():

    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def load_name_w2v(dataset_path):
    lines = []
    vecs = np.load(dataset_path)
    for vec in vecs:
        lines.append(vec)

    return lines

def load_dataset(dataset_path) -> list:

    lines = []
    try:
        with open(dataset_path, 'r') as file:
            for line in file.readlines():
                words = line.strip().split(' ')
                lines.append(words)
    except Exception:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                words = line.strip().split(' ')
                lines.append(words)
    return lines

def load_dataset_for_signature(dataset_path) -> list:

    lines = []
    try:
        with open(dataset_path, 'r') as file:
            for line in file.readlines():
                param_items = re.split('\(|\)', line.strip())
                one_line = []
                for param in param_items:
                    if param != '':
                        words = param.strip().split(' ')
                        one_line.append(words)
                lines.append(one_line)
    except Exception:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                param_items = re.split('\(|\)', line.strip())
                one_line = []
                for param in param_items:
                    if param != '':
                        words = param.strip().split(' ')
                        one_line.append(words)
                lines.append(one_line)
    return lines

def filter_data(codes, asts, nls, names):

    assert len(codes) == len(asts)
    assert len(asts) == len(nls)

    new_codes = []
    new_asts = []
    new_nls = []
    new_names = []
    for i in range(len(codes)):
        code = codes[i]
        ast = asts[i]
        nl = nls[i]
        name = names[i]
        if len(code) > config.max_code_length or len(nl) > config.max_nl_length or len(nl) < config.min_nl_length:
            continue
        new_codes.append(code)
        new_asts.append(ast)
        new_nls.append(nl)
        new_names.append(name)
    return new_codes, new_asts, new_nls, new_names


def init_vocab(name, lines, trim=False, min_count=None):

    vocab = Vocab(name)
    for line in lines:
        vocab.add_sentence(line)
    if trim:
        vocab.trim(min_count)
    return vocab


def init_decoder_inputs(batch_size, vocab: Vocab) -> torch.Tensor:

    return torch.tensor([vocab.word2index[_SOS]] * batch_size, device=config.device)


def filter_oov(inputs, vocab: Vocab):

    unk = vocab.word2index[_UNK]
    for index_step, step in enumerate(inputs):
        for index_word, word in enumerate(step):
            if word >= vocab.num_words:
                inputs[index_step][index_word] = unk
    return inputs


def get_seq_lens(batch: list) -> list:

    seq_lens = []
    for seq in batch:
        if len(seq) == 0:
            seq_lens.append(1)
        else:
            seq_lens.append(len(seq))
    return seq_lens

def get_seq_lens_for_signature(batch: list) -> list:

    seq_lens = []
    for line in batch:
        sentence_len = []
        for param in line:
            sentence_len.append(len(param))
        seq_lens.append(sentence_len)
    return seq_lens


def pad_one_batch(batch: list, vocab: Vocab) -> torch.Tensor:

    if len(max(*batch, key=lambda v: len(v))) != 0:
        batch = list(itertools.zip_longest(*batch, fillvalue=vocab.word2index[_PAD]))
        batch = [list(b) for b in batch]
    else:
        batch[0] = [vocab.word2index[_PAD]]
        batch = list(itertools.zip_longest(*batch, fillvalue=vocab.word2index[_PAD]))
        batch = [list(b) for b in batch]
    return torch.tensor(batch, device=config.device).long()



def indices_from_batch(batch: list, vocab: Vocab) -> list:

    indices = []
    for sentence in batch:
        indices_sentence = []
        for word in sentence:
            if word not in vocab.word2index:
                indices_sentence.append(vocab.word2index[_UNK])
            else:
                indices_sentence.append(vocab.word2index[word])
        indices_sentence.append(vocab.word2index[_EOS])
        indices.append(indices_sentence)
    return indices

def indices_from_batch_for_signature(batch: list, vocab: Vocab) -> list:

    indices = []
    for line in batch:
        params_indices = []
        for param in line:
            indices_sentence = []
            for word in param:
                if word not in vocab.word2index:
                    indices_sentence.append(vocab.word2index[_UNK])
                else:
                    indices_sentence.append(vocab.word2index[word])
            indices_sentence.append(vocab.word2index[_EOS])
            params_indices.append(indices_sentence)
        indices.append(params_indices)

    result = []
    mask_for_attention = []
    result.append([])
    result.append([])
    result.append([])
    result.append([])
    result.append([])
    for line in indices:
        line_mask = []
        i = 0
        while i < 5:
            if i < len(line):
                result[i].append(line[i])
                line_mask.append(1)
            else:
                result[i].append([])
                line_mask.append(0)
            i += 1
        mask_for_attention.append(line_mask)

    return result, mask_for_attention

def sort_batch(batch) -> (list, list, list):

    seq_lens = get_seq_lens(batch)
    pos = np.argsort(seq_lens)[::-1]
    batch = [batch[index] for index in pos]
    seq_lens.sort(reverse=True)
    return batch, seq_lens, pos


def restore_encoder_outputs(outputs: torch.Tensor, pos) -> torch.Tensor:

    rev_pos = np.argsort(pos)
    outputs = torch.index_select(outputs, 1, torch.tensor(rev_pos, device=config.device))
    return outputs


def get_pad_index(vocab: Vocab) -> int:
    return vocab.word2index[_PAD]


def get_sos_index(vocab: Vocab) -> int:
    return vocab.word2index[_SOS]


def get_eos_index(vocab: Vocab) -> int:
    return vocab.word2index[_EOS]


def collate_fn(batch, code_vocab, ast_vocab, nl_vocab, is_eval=False) -> \
        (torch.Tensor, list, list, torch.Tensor, list, list, torch.Tensor, list):

    batch = batch[0]
    code_batch = []
    ast_batch = []
    nl_batch = []
    for b in batch:
        code_batch.append(b[0])
        ast_batch.append(b[1])
        nl_batch.append(b[2])

    # transfer words to indices including oov words, and append EOS token to each sentence, list
    code_batch = indices_from_batch(code_batch, code_vocab)  # [B, T]
    ast_batch = indices_from_batch(ast_batch, ast_vocab)  # [B, T]
    if not is_eval:
        nl_batch = indices_from_batch(nl_batch, nl_vocab)  # [B, T]

    # sort each batch in decreasing order and get sequence lengths
    code_batch, code_seq_lens, code_pos = sort_batch(code_batch)
    ast_batch, ast_seq_lens, ast_pos = sort_batch(ast_batch)
    if not is_eval:
        nl_seq_lens = get_seq_lens(nl_batch)
    else:
        nl_seq_lens = None

    # pad and transpose, [T, B], tensor
    code_batch = pad_one_batch(code_batch, code_vocab)
    ast_batch = pad_one_batch(ast_batch, ast_vocab)
    if not is_eval:
        nl_batch = pad_one_batch(nl_batch, nl_vocab)

    return code_batch, code_seq_lens, code_pos, \
        ast_batch, ast_seq_lens, ast_pos, \
        nl_batch, nl_seq_lens

import ipdb
def unsort_collate_fn(batch, code_vocab, ast_vocab, nl_vocab, raw_nl=False):

    batch = batch[0]
    code_batch_tf = []
    ast_batch_tf = []
    name_batch_tf = []
    index_batch_tf = []

    code_batch_torch = []
    ast_batch_torch = []
    name_batch_torch = []
    index_batch_torch = []

    code_batch_neg = []
    ast_batch_neg = []
    name_batch_neg = []

    nl_batch = []
    for b in batch:
        code_batch_tf.append(b[0])
        ast_batch_tf.append(b[1])
        name_batch_tf.append(b[2])
        index_batch_tf.append(b[3])

        code_batch_torch.append(b[4])
        ast_batch_torch.append(b[5])
        name_batch_torch.append(b[6])
        index_batch_torch.append(b[7])

        code_batch_neg.append(b[8])
        ast_batch_neg.append(b[9])
        name_batch_neg.append(b[10])

        nl_batch.append(b[11])

    # transfer words to indices including oov words, and append EOS token to each sentence, list
    code_batch_tf = indices_from_batch(code_batch_tf, code_vocab)  # [B, T]
    code_batch_torch = indices_from_batch(code_batch_torch, code_vocab)  # [B, T]
    code_batch_neg = indices_from_batch(code_batch_neg, code_vocab)  # [B, T]

    ast_batch_tf, attention_mask_batch_tf = indices_from_batch_for_signature(ast_batch_tf, ast_vocab)  # [B, T]
    ast_batch_torch, attention_mask_batch_torch = indices_from_batch_for_signature(ast_batch_torch, ast_vocab)  # [B, T]
    ast_batch_neg, attention_mask_batch_neg = indices_from_batch_for_signature(ast_batch_neg, ast_vocab)  # [B, T]

    name_batch_tf = indices_from_batch(name_batch_tf, code_vocab)  # [B, T]
    name_batch_torch = indices_from_batch(name_batch_torch, code_vocab)  # [B, T]

    if not raw_nl:
        nl_batch = indices_from_batch(nl_batch, nl_vocab)  # [B, T]

    code_seq_lens_tf = get_seq_lens(code_batch_tf)
    code_seq_lens_torch = get_seq_lens(code_batch_torch)
    code_seq_lens_neg = get_seq_lens(code_batch_neg)

    name_seq_lens_tf = get_seq_lens(name_batch_tf)
    name_seq_lens_torch = get_seq_lens(name_batch_torch)


    ast_seq_lens_tf = []
    ast_seq_lens_tf.append(get_seq_lens(ast_batch_tf[0]))
    ast_seq_lens_tf.append(get_seq_lens(ast_batch_tf[1]))
    ast_seq_lens_tf.append(get_seq_lens(ast_batch_tf[2]))
    ast_seq_lens_tf.append(get_seq_lens(ast_batch_tf[3]))
    ast_seq_lens_tf.append(get_seq_lens(ast_batch_tf[4]))

    ast_seq_lens_torch = []
    ast_seq_lens_torch.append(get_seq_lens(ast_batch_torch[0]))
    ast_seq_lens_torch.append(get_seq_lens(ast_batch_torch[1]))
    ast_seq_lens_torch.append(get_seq_lens(ast_batch_torch[2]))
    ast_seq_lens_torch.append(get_seq_lens(ast_batch_torch[3]))
    ast_seq_lens_torch.append(get_seq_lens(ast_batch_torch[4]))

    ast_seq_lens_neg = []
    ast_seq_lens_neg.append(get_seq_lens(ast_batch_neg[0]))
    ast_seq_lens_neg.append(get_seq_lens(ast_batch_neg[1]))
    ast_seq_lens_neg.append(get_seq_lens(ast_batch_neg[2]))
    ast_seq_lens_neg.append(get_seq_lens(ast_batch_neg[3]))
    ast_seq_lens_neg.append(get_seq_lens(ast_batch_neg[4]))

    # ipdb.set_trace()
    nl_seq_lens = get_seq_lens(nl_batch)

    # pad and transpose, [T, B], tensor
    code_batch_tf = pad_one_batch(code_batch_tf, code_vocab)
    code_batch_torch = pad_one_batch(code_batch_torch, code_vocab)
    code_batch_neg = pad_one_batch(code_batch_neg, code_vocab)

    name_batch_tf = pad_one_batch(name_batch_tf, code_vocab)
    name_batch_torch = pad_one_batch(name_batch_torch, code_vocab)


    # ast_batch = []
    ast_batch_tf[0] = (pad_one_batch(ast_batch_tf[0], ast_vocab))
    ast_batch_tf[1] = (pad_one_batch(ast_batch_tf[1], ast_vocab))
    ast_batch_tf[2] = (pad_one_batch(ast_batch_tf[2], ast_vocab))
    ast_batch_tf[3] = (pad_one_batch(ast_batch_tf[3], ast_vocab))
    ast_batch_tf[4] = (pad_one_batch(ast_batch_tf[4], ast_vocab))

    ast_batch_torch[0] = (pad_one_batch(ast_batch_torch[0], ast_vocab))
    ast_batch_torch[1] = (pad_one_batch(ast_batch_torch[1], ast_vocab))
    ast_batch_torch[2] = (pad_one_batch(ast_batch_torch[2], ast_vocab))
    ast_batch_torch[3] = (pad_one_batch(ast_batch_torch[3], ast_vocab))
    ast_batch_torch[4] = (pad_one_batch(ast_batch_torch[4], ast_vocab))

    ast_batch_neg[0] = (pad_one_batch(ast_batch_neg[0], ast_vocab))
    ast_batch_neg[1] = (pad_one_batch(ast_batch_neg[1], ast_vocab))
    ast_batch_neg[2] = (pad_one_batch(ast_batch_neg[2], ast_vocab))
    ast_batch_neg[3] = (pad_one_batch(ast_batch_neg[3], ast_vocab))
    ast_batch_neg[4] = (pad_one_batch(ast_batch_neg[4], ast_vocab))

    if not raw_nl:
        nl_batch = pad_one_batch(nl_batch, nl_vocab)

    # name_batch_tf = torch.tensor(name_batch_tf, device=config.device).long()
    # name_batch_torch = torch.tensor(name_batch_torch, device=config.device).long()
    # name_batch_neg = torch.tensor(name_batch_neg, device=config.device).long()


    return code_batch_tf, code_seq_lens_tf, \
        ast_batch_tf, ast_seq_lens_tf, attention_mask_batch_tf, \
        name_batch_tf, name_seq_lens_tf, index_batch_tf,\
           code_batch_torch, code_seq_lens_torch, \
           ast_batch_torch, ast_seq_lens_torch, attention_mask_batch_torch, \
           name_batch_torch, name_seq_lens_torch, index_batch_torch,\
           code_batch_neg, code_seq_lens_neg, \
           ast_batch_neg, ast_seq_lens_neg, attention_mask_batch_neg, \
           name_batch_neg, \
           nl_batch, nl_seq_lens



def to_time(float_time):

    time_s = int(float_time)
    time_ms = int((float_time - time_s) * 1000)
    time_h = time_s // 3600
    time_s = time_s % 3600
    time_min = time_s // 60
    time_s = time_s % 60
    return time_h, time_min, time_s, time_ms


def print_train_progress(start_time, cur_time, epoch, n_epochs, index_batch,
                         batch_size, dataset_size, loss, last_print_index):
    spend = cur_time - start_time
    spend_h, spend_min, spend_s, spend_ms = to_time(spend)

    n_iter = (dataset_size + config.batch_size - 1) // config.batch_size
    len_epoch = len(str(n_epochs))
    len_iter = len(str(n_iter))
    percent_complete = (epoch / n_epochs +
                        (1 / n_epochs) / dataset_size * (index_batch * config.batch_size + batch_size)) * 100

    time_remaining = spend / percent_complete * (100 - percent_complete)
    remain_h, remain_min, remain_s, remain_ms = to_time(time_remaining)

    batch_length = index_batch - last_print_index
    if batch_length != 0:
        loss = loss / batch_length

    print('\033[0;36mtime\033[0m: {:2d}h {:2d}min {:2d}s {:3d}ms, '.format(
        spend_h, spend_min, spend_s, spend_ms), end='')
    print('\033[0;36mremaining\033[0m: {:2d}h {:2d}min {:2d}s {:3d}ms, '.format(
        remain_h, remain_min, remain_s, remain_ms), end='')
    print('\033[0;33mepoch\033[0m: %*d/%*d, \033[0;33mbatch\033[0m: %*d/%*d, ' %
          (len_epoch, epoch + 1, len_epoch, n_epochs, len_iter, index_batch, len_iter, n_iter - 1), end='')
    print('\033[0;32mpercent complete\033[0m: {:6.2f}%, \033[0;31mavg loss\033[0m: {:.4f}'.format(
        percent_complete, loss))

    config.logger.info('epoch: {}/{}, batch: {}/{}, avg loss: {:.4f}'.format(
        epoch + 1, n_epochs, index_batch, n_iter - 1, loss))


def plot_train_progress():
    pass


def is_unk(word):
    if word == _UNK:
        return True
    return False


def is_special_symbol(word):
    if word in _START_VOCAB:
        return True
    else:
        return False


