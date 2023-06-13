import itertools

import torchsummary
import torch
import torch.nn as nn
from models import Model
import utils
import config
import re
import wordninja
batch = [[1,2,3,4], [7,8,9]]
batch = list(itertools.zip_longest(*batch, fillvalue=0))
batch = [list(b) for b in batch]
print(batch)



# best_model_dict = torch.load('best_epoch-0_batch-last.pt', map_location=torch.device('cpu'))
#
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
#               is_eval=True)
# model.decoder = nn.Sequential()
#
# torch.save(model.state_dict(),'fine_tune_model.pt')