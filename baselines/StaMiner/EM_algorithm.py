import os
import shutil
import sys
import logging
import json
import operator
from pathlib import Path
from functools import reduce
from redbaron import RedBaron, IfelseblockNode, AtomtrailersNode, CallNode, NameNode, GetitemNode, NodeList, ForNode, \
    WhileNode, AssignmentNode, ReturnNode, PrintNode, LineProxyList, ProxyList, CommaProxyList, DotProxyList, \
    TryNode, CallArgumentNode, ClassNode, TupleNode, ListNode, CodeBlockNode, DefNode, DecoratorNode
from  pandas import read_csv
from math import isnan

torch_set = set()
tf_set = set()

def get_words(txt_path):
    control = ['if', 'for', 'while', 'try', 'elif', '']
    with open(txt_path, 'r') as r:
        str = r.read()
        words = str.split(' ,')
    result = []
    for word in words:
        if word not in control:
            result.append(word)
        # for i in key_words_torch:
        #     if word.find(i) is not -1:
        #         torch_set.add(word)
        #         break
        # for i in key_words_tf:
        #     if word.find(i) is not -1:
        #         tf_set.add(word)
        #         break
    return result

key_words_torch = ['torch.', 'nn.']
key_words_tf = ['tensorflow.', 'tf.']
key_words = []
key_words.extend(key_words_tf)
key_words.extend(key_words_torch)


TORCH_sentences = []
TF_sentences = []

# get_sentences()

def EM_Althgorth(TF_sentences, TORCH_sentences):
        print(len(TF_sentences))
        print(len(TORCH_sentences))

        f_word = list(set(reduce(operator.add, TORCH_sentences)))
        e_word = list(set(reduce(operator.add, TF_sentences)))
        T = {}

        for k in range(10):
            C = {}
            for m, l in zip(TORCH_sentences, TF_sentences):
                if k == 0:
                    for fi in m:
                        for ej in l:
                            if " % s| % s" % (fi, ej) not in T:
                                T["%s|% s" % (fi, ej)] = 1.0 / len(e_word)
                for i, fi in enumerate(m):
                    sum_t = sum([T["% s|% s" % (fi, ej)] for ej in l]) * 1.0
                    for j, ej in enumerate(l):
                        delta = T["% s|% s" % (fi, ej)] / sum_t
                        C[" % s % s" % (ej, fi)] = C.get(" % s % s" % (ej, fi), 0) + delta
                        C[" % s" % (ej)] = C.get(" % s" % (ej), 0) + delta
            # print("---iteration: % s---" % (k))
            for key in T:
                # print(key, ":", T[key])
                pass
            for f in f_word:
                for e in e_word:
                    if " % s % s" % (e, f) in C and " % s" % (e) in C:
                        T["% s|% s" % (f, e)] = C[" % s % s" % (e, f)] / C[" % s" % (e)]


        result = (sorted(T.items(), key=lambda item:item[1], reverse=True))
        with open(r'D:\Pycharm WorkSpace\\1126-pt2tf-10\\1126-pt2tf-10\\result_by_class.txt' , 'a') as writer:
            for i in result:
                api_pair = i[0]
                torch_api = api_pair.split('|')[0]
                tf_api = api_pair.split('|')[1]
                if torch_api == tf_api or tf_api == 'TF' + torch_api or (torch_api.startswith('self.') and tf_api.startswith('self.')):
                    continue
                if float(i[1]) <= 0.5:
                    break
                writer.write(str(i)+'\n')

        print(result)







