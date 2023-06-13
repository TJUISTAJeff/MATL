import torch
import torch.nn as nn
import torch.nn.functional
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import utils
import config
import data
import models
import eval


class Train(object):

    def __init__(self, vocab_file_path=None, model_file_path=None):

        # dataset
        self.train_dataset = data.CodePtrDataset(code_path_tf=config.train_code_path_tf,
                                                 ast_path_tf=config.train_sbt_path_tf,
                                                 code_path_torch=config.train_code_path_torch,
                                                 ast_path_torch=config.train_sbt_path_torch,
                                                 name_path_tf=config.train_name_path_tf,
                                                 name_path_torch=config.train_name_path_torch,
                                                 code_path_neg=config.train_code_path_neg,
                                                 ast_path_neg=config.train_sbt_path_neg,
                                                 name_path_neg=config.train_name_path_neg,
                                                 index_path_tf=config.train_index_path_tf,
                                                 index_path_torch=config.train_index_path_torch,
                                                 nl_path=config.train_nl_path)
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
            codes, asts, nls = self.train_dataset.get_dataset()
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
        # finetune_model = torch.load('preTrain_model/best_epoch-0_batch-last.pt', map_location=torch.device('cpu'))

        self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                  ast_vocab_size=self.ast_vocab_size,
                                  nl_vocab_size=self.nl_vocab_size,
                                  model_file_path='preTrain_model/hidden_size_128_back_up.pt')
        # print("Train self.model", self.model)
        self.params = list(self.model.code_encoder_tf.parameters()) + \
                      list(self.model.name_encoder_tf.parameters()) + \
                      list(self.model.ast_encoder_tf_0.parameters()) + \
            list(self.model.ast_encoder_tf_1.parameters()) + \
            list(self.model.ast_encoder_tf_2.parameters()) + \
            list(self.model.ast_encoder_tf_3.parameters()) + \
            list(self.model.ast_encoder_tf_4.parameters()) + \
            list(self.model.reduce_hidden_tf.parameters()) + \
          list(self.model.MLP_tf.parameters()) + \
          list(self.model.code_encoder_torch.parameters()) + \
                      list(self.model.name_encoder_torch.parameters()) + \
                      list(self.model.ast_encoder_torch_0.parameters()) + \
          list(self.model.ast_encoder_torch_1.parameters()) + \
          list(self.model.ast_encoder_torch_2.parameters()) + \
          list(self.model.ast_encoder_torch_3.parameters()) + \
          list(self.model.ast_encoder_torch_4.parameters()) + \
          list(self.model.reduce_hidden_torch.parameters()) + \
            list(self.model.MLP_torch.parameters())

            # optimizer
        self.optimizer = Adam([
            {'params': self.model.code_encoder_tf.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.name_encoder_tf.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.ast_encoder_tf_0.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_tf_1.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_tf_2.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_tf_3.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_tf_4.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.reduce_hidden_tf.parameters(), 'lr': config.reduce_hidden_lr},
            {'params': self.model.code_encoder_torch.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.name_encoder_torch.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.ast_encoder_torch_0.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_torch_1.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_torch_2.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_torch_3.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.ast_encoder_torch_4.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.reduce_hidden_torch.parameters(), 'lr': config.reduce_hidden_lr},
            # {'params': self.model.atten_tf.parameters(), 'lr': config.atten_lr},
            {'params': self.model.MLP_tf.parameters(), 'lr': config.MLP_lr},
            # {'params': self.model.atten_torch.parameters(), 'lr': config.atten_lr},
            {'params': self.model.MLP_torch.parameters(), 'lr': config.MLP_lr},

            
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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

        index_tf = batch[7]
        index_torch = batch[15]

        self.optimizer.zero_grad()

        _, _, decoder_hidden_tf, code_hidden_tf, signatures_hiddens_tf, \
        _, _, decoder_hidden_torch, code_hidden_torch, signatures_hiddens_torch, \
        _, _, decoder_hidden_neg \
            = self.model(batch, batch_size, self.nl_vocab)     # [T, B, nl_vocab_size]
        decoder_hidden_tf = decoder_hidden_tf.squeeze(0)
        decoder_hidden_torch = decoder_hidden_torch.squeeze(0)
        decoder_hidden_neg = decoder_hidden_neg.squeeze(0)


        loss =   criterion(decoder_hidden_tf, decoder_hidden_torch)

        # loss = 2 -  criterion(code_hidden_tf, code_hidden_torch) - criterion(signatures_hiddens_tf, signatures_hiddens_torch)




        # true_lable = torch.ones(len(decoder_hidden_tf),device='cuda:0' )
        # neg_lable = torch.full((len(decoder_hidden_tf),), fill_value=-1,device='cuda:0')
        # loss = 0.9 * criterion(decoder_hidden_tf, decoder_hidden_torch, true_lable) + 0.1 * criterion(decoder_hidden_tf, decoder_hidden_neg, neg_lable)
        loss = loss.mean()

        loss.backward()

        # address over fit
        torch.nn.utils.clip_grad_norm_(self.params, 5)

        self.optimizer.step()

        return loss, decoder_hidden_tf, decoder_hidden_torch, index_tf, index_torch, \
               code_hidden_tf, signatures_hiddens_tf, code_hidden_torch, signatures_hiddens_torch

    def train_iter(self):
        start_time = time.time()

        plot_losses = []

        # criterion = nn.functional.cosine_similarity
        criterion = nn.MSELoss()
        #criterion = nn.CrossEntropyLoss()

        for epoch in range(config.n_epochs):
            print_loss = 0
            plot_loss = 0
            last_print_index = 0
            last_plot_index = 0
            decoders_hidden_from = []
            decoders_hidden_to = []
            indexes_from = []
            indexes_to = []

            code_hiddens_from = []
            sig_hiddens_from = []
            code_hiddens_to = []
            sig_hiddens_to = []
            for index_batch, batch in enumerate(self.train_dataloader):
                # print(len(batch))
                batch_size = len(batch[0][0])

                loss ,decoder_hidden_from, decoder_hidden_to, index_from, index_to, \
                    code_hidden_from , sig_hidden_from ,code_hidden_to,sig_hidden_to \
                = self.train_one_batch(batch, batch_size, criterion)
                print_loss += loss.item()
                # print(loss.item())
                # print(loss)
                plot_loss += loss.item()

                decoders_hidden_from.append(decoder_hidden_from.cpu())
                decoders_hidden_to.append(decoder_hidden_to.cpu())
                indexes_from.append(index_from)
                indexes_to.append(index_to)

                code_hiddens_from.append(code_hidden_from)
                code_hiddens_to.append(code_hidden_to)
                sig_hiddens_from.append(sig_hidden_from)
                sig_hiddens_to.append(sig_hidden_to)

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
                    top1, top5 , top10 = calculate_metrix_one_batch(decoders_hidden_from, decoders_hidden_to, indexes_from, indexes_to, config.mapping_dict_path)
                    print('\nValidating the model at epoch {}, batch {} on valid dataset......'.format(
                        epoch, index_batch))
                    config.logger.info('Validating the model at epoch {}, batch {} on valid dataset.Top 1: {}, Top 5: {}, Top 10: {},'.format(
                        epoch, index_batch, top1, top5, top10))
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
        plt.plot(plot_losses)
        plt.savefig(os.path.join(config.out_dir, 'train_loss_{}.svg'.format(utils.get_timestamp())),
                    dpi=600, format='svg')
        utils.save_pickle(plot_losses, os.path.join(config.out_dir, 'plot_losses_{}.pk'.format(utils.get_timestamp())))

        # save the best model
        if config.save_best_model:
            best_model_name = 'best_epoch-{}_batch-{}.pt'.format(
                self.best_epoch_batch[0], self.best_epoch_batch[1] if self.best_epoch_batch[1] != -1 else 'last')
            self.save_model(name=best_model_name, state_dict=self.best_model)

    def save_model(self, name=None, state_dict=None):
        """
        save current model
        :param name: if given, name the model file by given name, else by current time
        :param state_dict: if given, save the given state dict, else save current model
        :return:
        """
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
        """
        get current state dict of model
        :return:
        """
        state_dict = {
                'code_encoder_tf': self.model.code_encoder_tf.state_dict(),
            'name_encoder_tf': self.model.name_encoder_tf.state_dict(),
                'ast_encoder_tf_0': self.model.ast_encoder_tf_0.state_dict(),
                'ast_encoder_tf_1': self.model.ast_encoder_tf_1.state_dict(),
                'ast_encoder_tf_2': self.model.ast_encoder_tf_2.state_dict(),
                'ast_encoder_tf_3': self.model.ast_encoder_tf_3.state_dict(),
                'ast_encoder_tf_4': self.model.ast_encoder_tf_4.state_dict(),
                'reduce_hidden_tf': self.model.reduce_hidden_tf.state_dict(),
                'code_encoder_torch': self.model.code_encoder_torch.state_dict(),
            'name_encoder_torch': self.model.name_encoder_torch.state_dict(),
                'ast_encoder_torch_0': self.model.ast_encoder_torch_0.state_dict(),
                'ast_encoder_torch_1': self.model.ast_encoder_torch_1.state_dict(),
                'ast_encoder_torch_2': self.model.ast_encoder_torch_2.state_dict(),
                'ast_encoder_torch_3': self.model.ast_encoder_torch_3.state_dict(),
                'ast_encoder_torch_4': self.model.ast_encoder_torch_4.state_dict(),
                'reduce_hidden_torch': self.model.reduce_hidden_torch.state_dict(),
                # 'attention_tf':self.model.atten_tf.state_dict(),
                'MLP_tf' : self.model.MLP_tf.state_dict(),
                # 'attention_torch': self.model.atten_torch.state_dict(),
                'MLP_torch': self.model.MLP_torch.state_dict(),
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


class MARLoss(nn.Module):
    def __init__(self):
        super(MARLoss, self).__init__()

    def forward(self, trained_vec, ground_truth, index_from, index_to, mapping_path):
        mapping = np.load(mapping_path, allow_pickle=True)
        mapping_dict = mapping.item()

        # import ipdb
        # ipdb.set_trace()
        def get_top(tensor):
            result_dict = {}
            for index_, to_hidden in zip(index_to, ground_truth):
                result_dict[index_] = torch.nn.functional.cosine_similarity(tensor, torch.unsqueeze(to_hidden, 0))

            result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)

            return result

        top_1 = 0
        top_5 = 0
        top_10 = 0

        query_num = 0
        hit_index = []
        for index, from_hidden in zip(index_from, trained_vec):
            if index == -1:
                continue
            else:
                query_num += 1
            tensor = torch.unsqueeze(from_hidden, 0)
            result = get_top(tensor)
            matched = False
            # print(index, '-----', result[0][0])
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

        # if not matched:
        #     print(index, '-------')
        #     print(result)
        MAR = 0
        for hit_ in hit_index:
            MAR += 1 / (hit_ + 1)
        MAR = MAR / query_num
        # print('Top 1: ', top_1 / query_num, "Top 5: ", top_5 / query_num, "Top 10: ", top_10 / query_num)
        return MAR

