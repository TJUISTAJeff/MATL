import os
import sys
import traceback
import numpy as np
import argparse
import threading
import codecs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch

from utils import normalize, similarity, sent2indexes
from data_loader import load_dict, load_vecs
import models, configs
  
codevecs, codebase = [], []

##### Data Set #####   
def load_codebase(code_path, chunk_size=60):
    """load codebase
      codefile: h5 file that stores raw code
    """
    logger.info(f'Loading codebase (chunk size={chunk_size})..')
    codebase= []
    codes = codecs.open(code_path, encoding='latin-1').readlines() # use codecs to read in case of encoding problem
    for i in range(0, len(codes), chunk_size):
        codebase.append(codes[i: i+chunk_size]) 
    '''
    import subprocess
    n_lines = int(subprocess.check_output(["wc", "-l", code_path], universal_newlines=True).split()[0])
    for i in range(1, n_lines+1, chunk_size):
        codecs = subprocess.check_output(["sed",'-n',f'{i},{i+chunk_size}p', code_path]).split()
        codebase.append(codecs)
   '''
    return codebase


### Results Data ###
def load_codevecs(vec_path, chunk_size=60):
    logger.debug(f'Loading code vectors (chunk size={chunk_size})..')       
    """read vectors (2D numpy array) from a hdf5 file"""
    codevecs=[]
    chunk_id = 0
    chunk_path = f"{vec_path[:-3]}_part{chunk_id}.h5"
    while os.path.exists(chunk_path):
        reprs = load_vecs(chunk_path)
        codevecs.append(reprs)
        chunk_id+=1
        chunk_path = f"{vec_path[:-3]}_part{chunk_id}.h5"
    #print(len(codevecs[0]))
    return codevecs

def search(config, model, vocab, query, n_results=10):
    model.eval()
    device = next(model.parameters()).device
    desc, desc_len =sent2indexes(query, vocab_desc, config['desc_len'])#convert query into word indices
    desc = torch.from_numpy(desc).unsqueeze(0).to(device)
    desc_len = torch.from_numpy(desc_len).clamp(max=config['desc_len']).to(device)
    with torch.no_grad():
        desc_repr = model.desc_encoding(desc, desc_len).data.cpu().numpy().astype(np.float32) # [1 x dim]
    if config['sim_measure']=='cos': # normalizing vector for fast cosine computation
        desc_repr = normalize(desc_repr) # [1 x dim]
    results =[]
    threads = []
    for i, codevecs_chunk in enumerate(codevecs):
        t = threading.Thread(target=search_thread, args = (results, desc_repr, codevecs_chunk, i, n_results, config['sim_measure']))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:#wait until all sub-threads have completed
        t.join()
    #print(len(results))
    return results

def search_thread(results, desc_repr, codevecs, i, n_results, sim_measure):        
#1. compute code similarities
    if sim_measure=='cos':
        chunk_sims = np.dot(codevecs, desc_repr.T)[:,0] # [pool_size]
    else:
        chunk_sims = similarity(codevecs, desc_repr, sim_measure) # [pool_size]
    
#2. select the top K results
    negsims = np.negative(chunk_sims)
    maxinds = np.argpartition(negsims, kth=n_results-1)
    #print('n_results----', n_results)
    maxinds = maxinds[:n_results]  
    #print(len(maxinds))
    chunk_codes = [codebase[i][k] for k in maxinds]
    chunk_sims = chunk_sims[maxinds]
    results.extend(zip(chunk_codes, chunk_sims))
    #print(len(results))
    
def postproc(codes_sims):
    codes_, sims_ = zip(*codes_sims)
    codes = [code for code in codes_]
    sims = [sim for sim in sims_]
    final_codes = []
    final_sims = []
    n = len(codes_sims)        
    for i in range(n):
        is_dup=False
        for j in range(i):
            if codes[i][:80]==codes[j][:80] and abs(sims[i]-sims[j])<0.01:
                is_dup=True
        if True:
            final_codes.append(codes[i])
            final_sims.append(sims[i])
    return zip(final_codes,final_sims)
    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='github', help='name of dataset.java, python')
    parser.add_argument('-t', '--timestamp', type=str, help='time stamp')
    parser.add_argument('--reload_from', type=int, default=-1, help='step to reload from')
    parser.add_argument('--chunk_size', type=int, default=60, help='codebase and code vector are stored in many chunks. '\
                         'Note: should be consistent with the same argument in the repr_code.py')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config = getattr(configs, 'config_'+args.model)()
    
    ##### Define model ######
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)#initialize the model
    ckpt=f'./output/{args.model}/{args.dataset}/{args.timestamp}/models/step{args.reload_from}.h5'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    data_path = args.data_path+args.dataset+'/'
    
    vocab_desc = load_dict(data_path+config['vocab_desc'])
    codebase = load_codebase(data_path+config['use_codebase'], args.chunk_size)
    codevecs = load_codevecs(data_path+config['use_codevecs'], args.chunk_size)
    print(len(codebase[0]), len(codevecs[0]))
    assert len(codebase)==len(codevecs), \
         "inconsistent number of chunks, check whether the specified files for codebase and code vectors are correct!"    
    
    with open('queries.txt', 'r') as reader:
        lines = reader.readlines()
    n_results = len(codebase[0])
    

    
    query_index = 0
    hit_5 = 0
    hit_1 = 0
    hit_10 = 0
    result_index = []
    import numpy as np
    query_idx = np.load('query.npy')
    mapping = np.load('data/github/mapping.npy', allow_pickle=True)
    for query in lines:
        query = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
        results = search(config, model, vocab_desc, query, n_results)
        #print(len(results))
        results = sorted(results, reverse=True, key=lambda x:x[1])
        #print(len(results))
        results = postproc(results)
        #print(len(results))
        results = list(results)
        #print(len(results))
        #print(len(results))
        #print('lines:', len(lines))
        for index in range(len(results)):
            if int(results[index][0]) in mapping.item()[query_idx[query_index]]:
                if index == 0:
                    hit_1 += 1
                    hit_5 += 1
                    hit_10 += 1
                elif index <= 4:
                    hit_5 += 1
                    hit_10 += 1
                elif index <= 9:
                    hit_10 += 1
                result_index.append(index)
                break

        results = '\n\n\n\n'.join(map(str,results)) #combine the result into a returning string
        #print(results)
        with open(f'result{query_index}.txt', 'w') as writer:
            writer.write(results)
        MAR = 0
        for index in result_index:
            MAR += 1/(index+1)

        MAR = MAR/33
        query_index += 1
    print(len(result_index),result_index)
    print(hit_1/33, hit_5/33, hit_10/33, MAR)
