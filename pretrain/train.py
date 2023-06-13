import torch
import torch.nn as nn
import torch.nn.functional
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import threading
import matplotlib.pyplot as plt

import utils
import config
import data
import models
import eval


class Train(object):

    def __init__(self, vocab_file_path=None, model_file_path=None):


        # dataset
        self.train_dataset = data.CodePtrDataset(code_path=config.train_code_path,
                                                 ast_path=config.train_sbt_path,
                                                 nl_path=config.train_nl_path,
                                                 name_path=config.train_name_path,
                                                 result_vec_path = config.train_result_path)
        self.train_dataset_size = len(self.train_dataset)
        # print(self.train_dataset_size)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                            code_vocab=self.code_vocab,
                                                                                            ast_vocab=self.ast_vocab,
                                                                                            nl_vocab=self.nl_vocab))

        # vocab
        self.code_vocab: utils.Vocab
        self.ast_vocab: utils.Vocab
        self.nl_vocab: utils.Vocab
        # load vocab from given path
        if vocab_file_path:
            code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
            self.code_vocab = utils.load_vocab_pk(code_vocab_path)
            self.ast_vocab = utils.load_vocab_pk(ast_vocab_path)
            self.nl_vocab = utils.load_vocab_pk(nl_vocab_path)
        # new vocab
        else:
            self.code_vocab = utils.Vocab('code_vocab')
            self.ast_vocab = utils.Vocab('ast_vocab')
            self.nl_vocab = utils.Vocab('nl_vocab')
            codes, asts, nls, names, _ = self.train_dataset.get_dataset()
            for code, ast, nl in zip(codes, asts, nls):
                self.code_vocab.add_sentence(code)
                self.ast_vocab.add_sentence(ast)
                self.nl_vocab.add_sentence(nl)

            self.origin_code_vocab_size = len(self.code_vocab)
            self.origin_nl_vocab_size = len(self.nl_vocab)

            # trim vocabulary
            self.code_vocab.trim(config.code_vocab_size)
            self.nl_vocab.trim(config.nl_vocab_size)
            # save vocabulary
            self.code_vocab.save(config.code_vocab_path)
            self.ast_vocab.save(config.ast_vocab_path)
            self.nl_vocab.save(config.nl_vocab_path)
            self.code_vocab.save_txt(config.code_vocab_txt_path)
            self.ast_vocab.save_txt(config.ast_vocab_txt_path)
            self.nl_vocab.save_txt(config.nl_vocab_txt_path)

        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab_size = len(self.nl_vocab)

        # model
        self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                  ast_vocab_size=self.ast_vocab_size,
                                  nl_vocab_size=self.nl_vocab_size,
                                  model_file_path=model_file_path)
        self.params = list(self.model.code_encoder.parameters()) + \
                      list(self.model.name_encoder.parameters()) + \
                      list(self.model.ast_encoder_0.parameters()) + \
                      list(self.model.ast_encoder_1.parameters()) + \
                      list(self.model.ast_encoder_2.parameters()) + \
                      list(self.model.ast_encoder_3.parameters()) + \
                      list(self.model.ast_encoder_4.parameters()) + \
                      list(self.model.reduce_hidden.parameters()) + \
                      list(self.model.atten.parameters()) + \
                      list(self.model.MLP.parameters())

        # optimizer
        self.optimizer = Adam([
            {'params': self.model.code_encoder.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.ast_encoder_0.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_1.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_2.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_3.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_4.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.reduce_hidden.parameters(), 'lr': config.reduce_hidden_lr},
            {'params': self.model.name_encoder.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.MLP.parameters(), 'lr': config.reduce_hidden_lr},
            {'params': self.model.atten.parameters(), 'lr': config.ast_encoder_lr},
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)

        if config.use_lr_decay:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=config.lr_decay_every,
                                                    gamma=config.lr_decay_rate)

        # best score and model(state dict)
        self.min_loss: float = 1000
        self.best_model: dict = {}
        self.best_epoch_batch: (int, int) = (None, None)

        # eval instance
        self.eval_instance = eval.Eval(self.get_cur_state_dict())

        # early stopping
        self.early_stopping = None
        if config.use_early_stopping:
            self.early_stopping = utils.EarlyStopping()

        config.model_dir = os.path.join(config.model_dir, utils.get_timestamp())
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)

    def run_train(self):

        self.train_iter()
        return self.best_model

    def train_one_batch(self, batch, batch_size, criterion):

        result_batch = batch[9]
        #import ipdb
        #ipdb.set_trace()
        #exit()

        #print(nl_batch.size())
        

        self.optimizer.zero_grad()

        _, _, decoder_hidden = self.model(batch, batch_size, self.nl_vocab)     # [T, B, nl_vocab_size]

        decoder_hidden = decoder_hidden.squeeze(0)

        #result_batch = result_batch.view(-1)
        
        decoder_hidden = decoder_hidden.to(torch.float)
        result_batch = result_batch.to(torch.float)

        #import ipdb
        #ipdb.set_trace()
        loss = criterion(decoder_hidden, result_batch)

        #loss = loss.mean()
        loss.backward()

        # address over fit
        #torch.nn.utils.clip_grad_norm_(self.params, 5)

        self.optimizer.step()

        return loss, decoder_hidden, result_batch

    def train_iter(self):
        start_time = time.time()

        plot_losses = []

        criterion = nn.MSELoss()

        for epoch in range(config.n_epochs):
            print_loss = 0
            plot_loss = 0
            last_print_index = 0
            last_plot_index = 0
            decoders_hidden = []
            results_ = []
            for index_batch, batch in enumerate(self.train_dataloader):

                batch_size = len(batch[0][0])

                loss, decoder_hidden, result_batch = self.train_one_batch(batch, batch_size, criterion)
                print_loss += loss
                plot_loss += loss

                decoders_hidden.append(decoder_hidden)
                results_.append(result_batch)
                # print train progress details
                if index_batch % config.print_every == 0:
                    cur_time = time.time()
                    utils.print_train_progress(start_time=start_time, cur_time=cur_time, epoch=epoch,
                                               n_epochs=config.n_epochs, index_batch=index_batch, batch_size=batch_size,
                                               dataset_size=self.train_dataset_size, loss=print_loss,
                                               last_print_index=last_print_index)
                    print_loss = 0
                    last_print_index = index_batch

                # plot train progress details
                if index_batch % config.plot_every == 0:
                    batch_length = index_batch - last_plot_index
                    if batch_length != 0:
                        plot_loss = plot_loss / batch_length
                    plot_losses.append(plot_loss)
                    plot_loss = 0
                    last_plot_index = index_batch

                # save check point
                if config.use_check_point and index_batch % config.save_check_point_every == 0:
                    pass

                # validate on the valid dataset every config.valid_every batches
                if config.validate_during_train and index_batch % config.validate_every == 0 and index_batch != 0:
                    #top1, top5 , top10 = calculate_metrix(decoders_hidden, results_)
                    print('\nValidating the model at epoch {}, batch {} on valid dataset......'.format(
                        epoch, index_batch))
                    config.logger.info('Validating the model at epoch {}, batch {} on valid dataset.'.format(
                        epoch, index_batch))
                    self.valid_state_dict(state_dict=self.get_cur_state_dict(), epoch=epoch, batch=index_batch)

                    if config.use_early_stopping:
                        if self.early_stopping.early_stop:
                            break
            if config.use_early_stopping:
                if self.early_stopping.early_stop:
                    break

            # validate on the valid dataset every epoch
            if config.validate_during_train:
                print('\nValidating the model at the end of epoch {} on valid dataset......'.format(epoch))
                config.logger.info('Validating the model at the end of epoch {} on valid dataset.'.format(epoch))
                self.valid_state_dict(self.get_cur_state_dict(), epoch=epoch)

                if config.use_early_stopping:
                    if self.early_stopping.early_stop:
                        break

            if config.use_lr_decay:
                self.lr_scheduler.step()

        plt.xlabel('every {} batches'.format(config.plot_every))
        plt.ylabel('avg loss')
        plt.plot(torch.tensor(plot_losses, device = 'cpu'))
        plt.savefig(os.path.join(config.out_dir, 'train_loss_{}.svg'.format(utils.get_timestamp())),
                    dpi=600, format='svg')
        utils.save_pickle(plot_losses, os.path.join(config.out_dir, 'plot_losses_{}.pk'.format(utils.get_timestamp())))

        # save the best model
        if config.save_best_model:
            best_model_name = 'best_epoch-{}_batch-{}.pt'.format(
                self.best_epoch_batch[0], self.best_epoch_batch[1] if self.best_epoch_batch[1] != -1 else 'last')
            self.save_model(name=best_model_name, state_dict=self.best_model)

    def save_model(self, name=None, state_dict=None):

        if state_dict is None:
            state_dict = self.get_cur_state_dict()
        if name is None:
            model_save_path = os.path.join(config.model_dir, 'model_{}.pt'.format(utils.get_timestamp()))
        else:
            model_save_path = os.path.join(config.model_dir, name)
        torch.save(state_dict, model_save_path)

    def save_check_point(self):
        pass

    def get_cur_state_dict(self) -> dict:

        state_dict = {
                'code_encoder': self.model.code_encoder.state_dict(),
                'name_encoder': self.model.name_encoder.state_dict(),
                'ast_encoder_0': self.model.ast_encoder_0.state_dict(),
                'ast_encoder_1': self.model.ast_encoder_1.state_dict(),
                'ast_encoder_2': self.model.ast_encoder_2.state_dict(),
                'ast_encoder_3': self.model.ast_encoder_3.state_dict(),
                'ast_encoder_4': self.model.ast_encoder_4.state_dict(),
                'reduce_hidden': self.model.reduce_hidden.state_dict(),
                'MLP':self.model.MLP.state_dict(),
                'Attention': self.model.atten.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        return state_dict

    def valid_state_dict(self, state_dict, epoch, batch=-1):
        self.eval_instance.set_state_dict(state_dict)
        loss = self.eval_instance.run_eval()

        if config.save_valid_model:
            model_name = 'model_valid-loss-{:.4f}_epoch-{}_batch-{}.pt'.format(loss, epoch, batch)
            save_thread = threading.Thread(target=self.save_model, args=(model_name, state_dict))
            save_thread.start()

        if loss < self.min_loss:
            self.min_loss = loss
            self.best_model = state_dict
            self.best_epoch_batch = (epoch, batch)

        if config.use_early_stopping:
            self.early_stopping(loss)


def calculate_metrix(trained_vec, ground_truth):
    tf_list = []
    torch_list = []

    for i in trained_vec:
        for j in i:
            tf_list.append(j)

    for i in ground_truth:
        for j in i:
            torch_list.append(j)

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
        while i < 5:
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
        while i < 10:
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
