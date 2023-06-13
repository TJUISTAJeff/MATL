import json
from tqdm import *
import h5py #导入python操作h5文件的库
import numpy as np
import tables

def h52txt(data_set_type):
    print('processing '+ data_set_type)

    table_name = tables.open_file('%s.apiseq.h5' % data_set_type)

    names = table_name.get_node('/phrases')[:].astype(np.long)
    idx_names = table_name.get_node('/indices')[:]

    voca_json = json.loads(open(r'vocab.apiseq.json', "r").read())

    for idx in tqdm(idx_names):
        with open(r'%s.apiseq.txt' % data_set_type, 'a') as writter:
            for token_id in names[idx[1]:idx[0]+idx[1]]:
                for token in voca_json:
                    if voca_json[token] == token_id:
                        writter.write(token +' ')
            writter.write('\n')

h52txt('test')
h52txt('train')
h52txt('use')






