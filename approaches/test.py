import torchsummary
import torch
import torch.nn as nn
from models import Model
import utils
import config



best_model_dict = torch.load('model_valid-loss-5.4659_epoch-0_batch--1.pt', map_location=torch.device('cpu'))

# code_vocab = utils.load_vocab_pk(config.code_vocab_path)
# code_vocab_size = len(code_vocab)
# ast_vocab = utils.load_vocab_pk(config.ast_vocab_path)
# ast_vocab_size = len(ast_vocab)
# nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
# nl_vocab_size = len(nl_vocab)
#
# model = Model(code_vocab_size=code_vocab_size,
#               ast_vocab_size=ast_vocab_size,
#               nl_vocab_size=nl_vocab_size,
#               model_state_dict=best_model_dict,
#               is_eval=False)
# model.decoder = nn.Sequential()
for child in best_model_dict:
    print(child)
# print(best_model_dict)

# torch.save(model.state_dict(),'fine_tune_model.pt')
