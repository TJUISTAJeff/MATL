import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional
import os
import time
import numpy as np
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
        self.dataset = data.CodePtrDataset(code_path=config.valid_code_path,
                                           ast_path=config.valid_sbt_path,
                                           nl_path=config.valid_nl_path,
                                           name_path=config.valid_name_path,
                                           result_vec_path=config.valid_result_path)
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

            # code_batch and ast_batch: [T, B]
            # nl_batch is raw data, [B, T] in list
            # nl_seq_lens is None
            result_batch = batch[9]

            _, _, decoder_hidden = self.model(batch, batch_size, self.nl_vocab)  # [T, B, nl_vocab_size]

            decoder_hidden = decoder_hidden.squeeze(0)
            #result_batch = result_batch.view(-1)

            loss = criterion(decoder_hidden, result_batch)

            return loss, decoder_hidden, result_batch

    def eval_iter(self):

        epoch_loss = 0
        criterion = nn.MSELoss()
        decoders_hidden = []
        results_ = []

        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch[0].shape[1]

            loss ,decoder_hidden, result_batch = self.eval_one_batch(batch, batch_size, criterion=criterion)
            epoch_loss += loss

            decoders_hidden.append(decoder_hidden)
            results_.append(result_batch)

        avg_loss = epoch_loss / len(self.dataloader)

        #top1, top5, top10 = calculate_metrix(decoders_hidden, results_)
        print('Validate completed, avg loss: {:.4f}.\n'.format(avg_loss))
        config.logger.info('Validate completed, avg loss: {:.4f}, '.format(avg_loss))

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
        self.dataset = data.CodePtrDataset(code_path=config.test_code_path,
                                           ast_path=config.test_sbt_path,
                                           nl_path=config.test_nl_path,
                                           name_path=config.test_name_path,
                                           result_vec_path=config.test_result_path)
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

        decoders_hidden, result = self.test_iter()
        
        np.save('decoders_hidden.npy', decoders_hidden)
        np.save('result.npy', result)


        #calculate_metrix(decoders_hidden, result)

        return decoders_hidden, result

    def test_one_batch(self, batch, batch_size):

        with torch.no_grad():
            result_batch = batch[9]

            # outputs: [T, B, H]
            # hidden: [1, B, H]
            _, _, decoder_hidden \
                = self.model(batch, batch_size, self.nl_vocab, is_test=True)  # [T, B, nl_vocab_size]

            return decoder_hidden, result_batch

    def test_iter(self):

        start_time = time.time()
        decoders_hidden = []
        results_ = []

        out_file = None
        if config.save_test_details:
            try:
                out_file = open(os.path.join(config.out_dir, 'test_details_{}.txt'.format(utils.get_timestamp())),
                                encoding='utf-8',
                                mode='w')
            except IOError:
                print('Test details file open failed.')

        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch[0].shape[1]

            decoder_hidden, result_batch = self.test_one_batch(batch, batch_size)
            decoders_hidden.append(decoder_hidden.cpu())
            results_.append(result_batch.cpu())

        return decoders_hidden, results_


def calculate_metrix(trained_vec, ground_truth):
    tf_list = []
    torch_list = []

    for i in trained_vec:
        for j in i[0]:
            tf_list.append([j][0])

    for i in ground_truth:
        for j in i[0]:
            torch_list.append([j][0])
    #import ipdb
    #ipdb.set_trace()
    def get_top(tensor):
        result_dict = {}
        for i in range(len(torch_list)):
            result_dict[i] = torch.nn.functional.cosine_similarity(tensor, torch.unsqueeze(torch_list[i], 0))

        result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)

        return result

    top_1 = 0
    top_5 = 0
    top_10 = 0
    for index in range(len(tf_list)):
        tensor = torch.unsqueeze(tf_list[index], 0)
        result = get_top(tensor)
        matched = False
        #print(index, '-----', result[0][0])
        if result[0][0] == index:
            matched = True
            top_1 += 1
            top_5 += 1
            top_10 += 1
            continue
        i = 0
        while i < 5 and not matched:
            if i == 0:
                i += 1
                continue
            if result[i][0] == index:
                matched = True
                top_5 += 1
                top_10 += 1
                break
            i += 1
        
        i = 0
        while i < 10 and not matched:
            if i < 5:
                i += 1
                continue
            if result[i][0] == index:
                matched = True
                top_10 += 1
                break
            i += 1

    # if not matched:
    #     print(index, '-------')
    #     print(result)

    print('Top 1: ', top_1 / len(tf_list), "Top 5: ", top_5 / len(tf_list), "Top 10: ", top_10 / len(tf_list))
    return top_1 / len(tf_list), top_5 / len(tf_list), top_10 / len(tf_list)
