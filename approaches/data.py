from torch.utils.data import Dataset
import numpy as np
import utils


class CodePtrDataset(Dataset):

    def __init__(self, code_path_tf, ast_path_tf, name_path_tf, index_path_tf, code_path_torch, ast_path_torch, name_path_torch, index_path_torch, code_path_neg, ast_path_neg, name_path_neg, \
                 nl_path):
        # get lines
        codes_tf = utils.load_dataset(code_path_tf)
        asts_tf = utils.load_dataset_for_signature(ast_path_tf)
        names_tf = utils.load_dataset(name_path_tf)
        if index_path_tf is not None:
            index_tf = np.load(index_path_tf)
        else:
            index_tf = None

        codes_torch = utils.load_dataset(code_path_torch)
        asts_torch = utils.load_dataset_for_signature(ast_path_torch)
        names_torch = utils.load_dataset(name_path_torch)
        # names_torch = utils.load_name_w2v(name_path_torch)
        if index_path_torch is not None:
            index_torch = np.load(index_path_torch)
        else:
            index_torch = None

        nls = utils.load_dataset(nl_path)

        if not (len(codes_tf) == len(asts_tf) and len(codes_tf) == len(asts_tf) and len(asts_torch) == len(codes_torch) ):
            print(len(codes_tf), len(asts_tf), len(codes_torch), len(asts_torch))
            raise Exception('The lengths of three dataset do not match.')

        if len(codes_tf) < len(codes_torch):
            offset_num = len(asts_torch) - len(asts_tf)
            codes_tf.extend(codes_torch[len(codes_tf):(len(codes_torch))])
            asts_tf.extend(codes_torch[len(asts_tf):(len(asts_torch))])
            for offset in range(offset_num):
                index_tf = np.append(index_tf, -1)
        print(len(index_tf), len(index_torch), len(codes_tf), len(codes_torch), len(asts_tf), len(asts_torch))
        print("from num : " , len(codes_tf), 'to num : ', len(codes_torch))


        if code_path_neg is not None:
            codes_neg = utils.load_dataset(code_path_neg)
            asts_neg = utils.load_dataset_for_signature(ast_path_neg)
        else:
            codes_neg = codes_tf
            asts_neg = asts_tf

        self.codes_tf, self.asts_tf, self.names_tf, self.index_tf, self.codes_torch, self.asts_torch, self.names_torch, self.index_torch, self.codes_neg, self.asts_neg, self.names_neg, self.nls = \
            (codes_tf, asts_tf,      names_tf,      index_tf,      codes_torch,      asts_torch,      names_torch,      index_torch,      codes_neg,      asts_neg,      None,      nls)
        # print(self.codes)

    def __len__(self):
        return len(self.codes_tf)

    def __getitem__(self, index):
        if self.index_torch is not None:
            return self.codes_tf[index], self.asts_tf[index], self.names_tf[index], self.index_tf[index], \
                   self.codes_torch[index], self.asts_torch[index], self.names_torch[index], self.index_torch[index],\
                   self.codes_neg[index], self.asts_neg[index], None, \
                   self.nls[index]
        else:
            return self.codes_tf[index], self.asts_tf[index], self.names_tf[index], None, \
                   self.codes_torch[index], self.asts_torch[index], self.names_torch[index], None, \
                   self.codes_neg[index], self.asts_neg[index], None, \
                   self.nls[index]


    def get_dataset(self):
        return self.codes_tf, self.asts_tf, self.names_tf, self.index_tf, \
               self.codes_torch, self.asts_torch,  self.names_torch, self.index_torch, \
               self.codes_neg, self.asts_neg, None, \
               self.nls
