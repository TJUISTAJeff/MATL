import os

import numpy as np
import torch

import config
import train
import eval


def _train(vocab_file_path=None, model_file_path=None):
    print('\nStarting the training process......\n')

    if vocab_file_path:
        code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
        print('Vocabulary will be built by given file path.')
        print('\tsource code vocabulary path:\t', os.path.join(config.vocab_dir, code_vocab_path))
        print('\tast of code vocabulary path:\t', os.path.join(config.vocab_dir, ast_vocab_path))
        print('\tcode comment vocabulary path:\t', os.path.join(config.vocab_dir, nl_vocab_path))
    else:
        print('Vocabulary will be built according to dataset.')

    if model_file_path:
        print('Model will be built by given state dict file path:', os.path.join(config.model_dir, model_file_path))
    else:
        print('Model will be created by program.')

    print('\nInitializing the training environments......\n')
    train_instance = train.Train(vocab_file_path=vocab_file_path, model_file_path=model_file_path)
    print('Environments built successfully.\n')
    print('Size of train dataset:', train_instance.train_dataset_size)


    if config.validate_during_train:
        print('\nValidate every', config.validate_every, 'batches and each epoch.')
        print('Size of validation dataset:', train_instance.eval_instance.dataset_size)
        config.logger.info('Size of validation dataset: {}'.format(train_instance.eval_instance.dataset_size))

    print('\nStart training......\n')
    config.logger.info('Start training.')
    best_model = train_instance.run_train()
    print('\nTraining is done.')
    config.logger.info('Training is done.')

    return best_model


def _test(model):
    print('\nInitializing the test environments......')
    test_instance = eval.Test(model)
    print('Environments built successfully.\n')
    print('Size of test dataset:', test_instance.dataset_size)
    config.logger.info('Size of test dataset: {}'.format(test_instance.dataset_size))

    config.logger.info('Start Testing.')
    print('\nStart Testing......')
    print('Testing is done.')


if __name__ == '__main__':
    with torch.cuda.device(1):
        best_model_dict = _train(vocab_file_path=('code_vocab.pk', 'ast_vocab.pk', 'nl_vocab.pk'))
        #best_model_dict = torch.load('model/20221212_162029/model_valid-loss-0.0000_epoch-27_batch--1.pt')
        _test(best_model_dict)
        # _test(os.path.join('20200521_203654', 'best_epoch-1_batch-last.pt'))
