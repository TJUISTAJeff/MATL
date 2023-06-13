import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time

import models
import data
import utils
import config


class Eval(object):

    def __init__(self, model):

        # vocabulary
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab = utils.load_vocab_pk(config.ast_vocab_path)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)

        # dataset
        self.dataset = data.CodePtrDataset(code_path_tf=config.valid_code_path_tf,
                                             ast_path_tf=config.valid_sbt_path_tf,
                                             code_path_torch=config.valid_code_path_torch,
                                             ast_path_torch=config.valid_sbt_path_torch,
                                             name_path_tf=config.valid_name_path_tf,
                                             name_path_torch=config.valid_name_path_torch,
                                           code_path_neg=config.valid_code_path_neg,
                                           ast_path_neg=config.valid_sbt_path_neg,
                                           name_path_neg=config.valid_name_path_neg,
                                           index_path_tf=config.valid_index_path_tf,
                                           index_path_torch=config.valid_index_path_torch,
                                           nl_path=config.valid_nl_path)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=config.eval_batch_size,
                                     collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                      code_vocab=self.code_vocab,
                                                                                      ast_vocab=self.ast_vocab,
                                                                                      nl_vocab=self.nl_vocab))

        # model
        if isinstance(model, str):
            self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_file_path=os.path.join(config.model_dir, model),
                                      is_eval=True)
        elif isinstance(model, dict):
            self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_state_dict=model,
                                      is_eval=True)
        else:
            raise Exception('Parameter \'model\' for class \'Eval\' must be file name or state_dict of the model.')

    def run_eval(self):
        loss = self.eval_iter()
        return loss

    def eval_one_batch(self, batch, batch_size, criterion):
        with torch.no_grad():
            index_tf = batch[7]
            index_torch = batch[15]
            # code_batch and ast_batch: [T, B]
            # nl_batch is raw data, [B, T] in list
            # nl_seq_lens is None

            _, _, decoder_hidden_tf, code_hidden_tf, signatures_hiddens_tf, \
            _, _, decoder_hidden_torch, code_hidden_torch, signatures_hiddens_torch, \
            _, _, decoder_hidden_neg \
                = self.model(batch, batch_size, self.nl_vocab)   # [T, B, nl_vocab_size]

            decoder_hidden_tf = decoder_hidden_tf.squeeze(0)
            decoder_hidden_torch = decoder_hidden_torch.squeeze(0)
            decoder_hidden_neg = decoder_hidden_neg.squeeze(0)
 

            loss =  criterion(decoder_hidden_tf, decoder_hidden_torch)
            # loss = 2 - criterion(code_hidden_tf, code_hidden_torch) - criterion(signatures_hiddens_tf,
            #                                                                     signatures_hiddens_torch)

            loss = loss.mean()


            return loss, decoder_hidden_tf, decoder_hidden_torch, index_tf, index_torch, \
                code_hidden_tf, signatures_hiddens_tf, code_hidden_torch, signatures_hiddens_torch

    def eval_iter(self):

        epoch_loss = 0
        # criterion = nn.functional.cosine_similarity
        criterion = nn.MSELoss()
        #criterion = nn.CrossEntropyLoss()

        decoders_hidden_from = []
        decoders_hidden_to = []
        indexes_from = []
        indexes_to = []

        hiddens_from = []
        hiddens_to = []


        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch[0].shape[1]

            loss ,decoder_hidden_from, decoder_hidden_to, index_batch_from, index_batch_to,\
                code_hidden_from , sig_hidden_from ,code_hidden_to,sig_hidden_to \
                = self.eval_one_batch(batch, batch_size, criterion=criterion)
            epoch_loss += loss.item()

            decoders_hidden_to.append(decoder_hidden_to)
            decoders_hidden_from.append(decoder_hidden_from)
            indexes_from.append(index_batch_from)
            indexes_to.append(index_batch_to)

            hiddens_from.append(torch.squeeze(torch.cat((code_hidden_from, sig_hidden_from), 2), 0))
            hiddens_to.append(torch.squeeze(torch.cat((code_hidden_to, sig_hidden_to), 2), 0))

        avg_loss = epoch_loss / len(self.dataloader)
        # print('Avg_loss:' , avg_loss)

        # import ipdb
        # ipdb.set_trace()

        top1, top5, top10 = calculate_metrix(decoders_hidden_from, decoders_hidden_to, indexes_from, indexes_to, config.mapping_dict_path)
        # top1, top5, top10 = calculate_metrix(hiddens_from, hiddens_to, indexes_from, indexes_to, config.mapping_dict_path)

        print('Validate completed, avg loss: {:.4f}.\n'.format(avg_loss))
        config.logger.info('Validate completed, avg loss: {:.4f}. Top 1: {:.4f}, Top 5: {:.4f} ,Top 10: {:.4f}'.format(avg_loss, top1, top5, top10))
        return avg_loss

    def set_state_dict(self, state_dict):
        self.model.set_state_dict(state_dict)


class Test(object):

    def __init__(self, model):

        # vocabulary
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab = utils.load_vocab_pk(config.ast_vocab_path)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)

        # dataset
        self.dataset = data.CodePtrDataset(code_path_tf=config.test_code_path_tf,
                                             ast_path_tf=config.test_sbt_path_tf,
                                             code_path_torch=config.test_code_path_torch,
                                             ast_path_torch=config.test_sbt_path_torch,
                                             name_path_tf=config.test_name_path_tf,
                                             name_path_torch=config.test_name_path_torch,
                                           code_path_neg=None,
                                           ast_path_neg=None,
                                           name_path_neg=None,
                                           index_path_tf=config.test_index_path_tf,
                                           index_path_torch=config.test_index_path_torch,
                                           nl_path=config.test_nl_path)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=config.test_batch_size,
                                     collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                      code_vocab=self.code_vocab,
                                                                                      ast_vocab=self.ast_vocab,
                                                                                      nl_vocab=self.nl_vocab,
                                                                                      raw_nl=True))

        # model
        if isinstance(model, str):
            self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_file_path=os.path.join(config.model_dir, model),
                                      is_eval=True)
        elif isinstance(model, dict):
            self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_state_dict=model,
                                      is_eval=True)
        else:
            raise Exception('Parameter \'model\' for class \'Test\' must be file name or state_dict of the model.')

    def run_test(self) -> dict:
        decoders_hidden_tf, decoders_hidden_torch, indexes_from, indexes_to ,\
            hiddens_from, hiddens_to = self.test_iter()



        import numpy as np

        np.save('decoders_hidden_tf.npy', decoders_hidden_tf)
        np.save('decoders_hidden_torch.npy', decoders_hidden_torch)
        np.save('indexes_from.npy', indexes_from)
        np.save('indexes_to.npy', indexes_to)

        calculate_metrix(decoders_hidden_tf, decoders_hidden_torch, indexes_from, indexes_to, config.mapping_dict_path, is_test=True)
        # calculate_metrix(hiddens_from, hiddens_to, indexes_from, indexes_to, config.mapping_dict_path)


        return decoders_hidden_tf, decoders_hidden_torch

    def test_one_batch(self, batch, batch_size):
        with torch.no_grad():
            index_tf = batch[7]
            index_torch = batch[15]

            # outputs: [T, B, H]
            # hidden: [1, B, H]
            _, _, decoder_hidden_tf, code_hidden_tf, signatures_hiddens_tf, \
            _, _, decoder_hidden_torch, code_hidden_torch, signatures_hiddens_torch, \
            _, _, decoder_hidden_neg \
                = self.model(batch, batch_size, self.nl_vocab, is_test = True)  # [T, B, nl_vocab_size]
            
            decoder_hidden_tf = torch.squeeze(decoder_hidden_tf, 0)
            decoder_hidden_torch = torch.squeeze(decoder_hidden_torch, 0)

            return decoder_hidden_tf, decoder_hidden_torch, index_tf, index_torch, \
                    code_hidden_tf, signatures_hiddens_tf, code_hidden_torch, signatures_hiddens_torch

    def test_iter(self):


        decoders_hidden_tf = []
        decoders_hidden_torch = []
        indexes_from = []
        indexes_to = []

        hiddens_from = []
        hiddens_to = []


        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch[0].shape[1]

            decoder_hidden_tf, decoder_hidden_torch, index_from, index_to ,\
                code_hidden_from , sig_hidden_from ,code_hidden_to,sig_hidden_to \
                = self.test_one_batch(batch, batch_size)
            

            decoders_hidden_tf.append(decoder_hidden_tf.cpu())
            decoders_hidden_torch.append(decoder_hidden_torch.cpu())
            indexes_from.append(index_from)
            indexes_to.append(index_to)

            hiddens_from.append(torch.squeeze(torch.cat((code_hidden_from, sig_hidden_from), 2), 0))
            hiddens_to.append(torch.squeeze(torch.cat((code_hidden_to, sig_hidden_to), 2), 0))


        return decoders_hidden_tf, decoders_hidden_torch, indexes_from, indexes_to, hiddens_from, hiddens_to


def calculate_metrix(trained_vec, ground_truth, index_from, index_to, mapping_path, is_test = False):
    mapping = np.load(mapping_path, allow_pickle=True)
    mapping_dict = mapping.item()

    tf_list = []
    torch_list = []
    indexes_from = []
    indexes_to = []

    for i in trained_vec:
        for j in i:
            tf_list.append([j][0])

    for i in ground_truth:
        for j in i:
            torch_list.append([j][0])

    for i in index_from:
        for j in i:
            indexes_from.append(j)

    for i in index_to:
        for j in i:
            indexes_to.append(j)

    # import ipdb
    # ipdb.set_trace()

    def get_top(tensor):
        result_dict = {}
        for index_, to_hidden in zip(indexes_to, torch_list):

            # result_dict[index_] = torch.nn.functional.cosine_similarity(tensor, torch.unsqueeze(to_hidden, 0))
            result_dict[index_] = 1 - torch.nn.functional.mse_loss(tensor, torch.unsqueeze(to_hidden, 0))
            #result_dict[index_] = 1 - torch.nn.functional.cross_entropy(tensor, torch.unsqueeze(to_hidden, 0))
        result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)

        return result

    top_1 = 0
    top_5 = 0
    top_10 = 0

    query_num = 0
    hit_index = []
    for index, from_hidden in zip(indexes_from, tf_list):
        if index == -1:
            continue
        else:
            query_num += 1
        tensor = torch.unsqueeze(from_hidden, 0)
        result = get_top(tensor)
        matched = False

        if is_test:
            print('index: ', index, ' result: ', result[0][0], ' ground_truth: ', mapping_dict[index])
        if result[0][0] in mapping_dict[index]:
            matched = True
            top_1 += 1
            top_5 += 1
            top_10 += 1
            hit_index.append(0)
            continue
        i = 0
        while i < 5 and not matched:
            if i == 0:
                i += 1
                continue
            if result[i][0] in mapping_dict[index]:
                matched = True
                top_5 += 1
                top_10 += 1
                hit_index.append(i)
                break
            i += 1

        i = 0
        while i < 10 and not matched:
            if i < 5:
                i += 1
                continue
            if result[i][0] in mapping_dict[index]:
                matched = True
                top_10 += 1
                hit_index.append(i)
                break
            i += 1

    MAR = 0
    for hit_ in hit_index:
        MAR += 1/(hit_ + 1)
    MAR = MAR/query_num
    print('Top 1: ', top_1 / query_num, "Top 5: ", top_5 / query_num, "Top 10: ", top_10 / query_num, 'MAR : ', MAR)
    return top_1 / query_num, top_5 / query_num, top_10 / query_num
