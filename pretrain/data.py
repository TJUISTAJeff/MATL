from torch.utils.data import Dataset
import numpy as np
import utils


class CodePtrDataset(Dataset):

    def __init__(self, code_path, ast_path, nl_path, name_path, result_vec_path):
        # get lines
        codes = utils.load_dataset(code_path)
        asts = utils.load_dataset(ast_path)
        nls = utils.load_dataset(nl_path)
        names = utils.load_dataset(name_path)
        result_vecs = np.load(result_vec_path)

        #import ipdb
        #ipdb.set_trace()

        if len(codes) != len(asts) or len(codes) != len(nls) or len(asts) != len(nls)  or len(asts) != len(result_vecs):
            print(len(asts), len(result_vecs))
            raise Exception('The lengths of three dataset do not match.')


        self.codes, self.asts, self.nls, self.names, self.result_vecs = utils.filter_data(codes, asts, nls, names, result_vecs)
        # print(self.codes)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.codes[index], self.asts[index], self.nls[index],  self.names[index], self.result_vecs[index]

    def get_dataset(self):
        return self.codes, self.asts, self.nls, self.names, self.result_vecs
