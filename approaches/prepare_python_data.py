import json
import re
import nltk.stem as ns
lemmatizer = ns.WordNetLemmatizer()

def std_signature(signature):
    lines = signature.splitlines()
    char_num = 0
    for char in lines[0]:
        if char == ' ':
            char_num += 1
        else:
            break
    signature_std = ''
    for line in lines:
        signature_std += line[char_num:] + '\n'
    return (signature_std)

def get_code_token(code):
    result_str = []
    tmp = re.split('([\[\]{}()| \-,.:;!@#$%^=&*_+\n])', code)
    for word in tmp:
        if word != '\n':
            result_str.append(lemmatizer.lemmatize(word.lower(), pos='v'))
    # print(result_str)
    return ' '.join(result_str)

def get_desc_token(desc):
    result_str = []
    tmp = re.split('([\[\]{}()| \-,.:;!@#$%^=&*_+\n])', desc)
    for word in tmp:
        if word != '\n':
            result_str.append(lemmatizer.lemmatize(word.lower(), pos='v'))
    # print(result_str)
    return ' '.join(result_str)

def get_signature_token(signature):
    lines = signature.splitlines()
    result = ''
    for index in range(len(lines)):
        if lines[index].strip() == 'Parameters' or lines[index].strip().startswith('---'):
            continue
        if not lines[index].startswith(' '):
            para = lines[index].strip().split(':', 1)[0]
            result_str = []
            tmp = re.split('[\[\]{}()| \-,.:;!@#$%^=&*_+\n]', lines[index].strip().split(':', 1)[-1])
            result += ' ( ' + para + " : "
            for word in tmp:
                if word != '\n':
                    result_str.append(lemmatizer.lemmatize(word.lower(), pos='v'))
            # print(result_str)

            result +=' '.join(result_str)


        else:
            result_str = []
            tmp = re.split('[\[\]{}()| \-,.:;!@#$%^=&*_+\n]', lines[index].strip().split(':', 1)[-1])
            for word in tmp:
                if word != '\n':
                    result_str.append(lemmatizer.lemmatize(word.lower(), pos='v'))
            # print(result_str)
            result += ' '
            result += ' '.join(result_str)
            if index+1 == len(lines):
                result += ' )'
            elif not lines[index+1].startswith(' '):
                # print(11111)
                result += ' )'

    return result

index = 0
dict = json.load(open(r'python_4w_data/train_data.json', 'r'))
for i in dict:
    if index < 40000:
        index += 1
        continue
    signature = dict[i]['signature']
    # print(get_code_token(dict[i]['source']))
    sig = (get_signature_token(signature))
    if 'desc' in dict[i]:
        desc = (get_desc_token(dict[i]['desc']))
    else:
        desc = ''
    source = (get_code_token(dict[i]['source']))
    if index < 40000:
        with open(r'train.token.code', 'a') as wr:
            wr.write(desc +'\n')
        with open(r'train.token.nl', 'a', encoding='utf-8') as wr:
            wr.write(source +'\n')

        with open(r'train.token.sbt', 'a', encoding='utf-8') as wr:
            wr.write(sig +'\n')

    elif index < 43000:
        with open(r'test.token.code', 'a') as wr:
            wr.write(desc +'\n')

        with open(r'test.token.nl', 'a', encoding='utf-8') as wr:
            wr.write(source + '\n')


        with open(r'test.token.sbt', 'a', encoding='utf-8') as wr:
            wr.write(sig +'\n')

    else:
        with open(r'valid.token.code', 'a') as wr:
            wr.write(desc +'\n')

        with open(r'valid.token.nl', 'a', encoding='utf-8') as wr:
            wr.write(source + '\n')


        with open(r'valid.token.sbt', 'a', encoding='utf-8') as wr:
            wr.write(sig +'\n')



    index += 1


print(len(dict))
