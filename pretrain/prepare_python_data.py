import json
import re
import wordninja
import nltk.stem as ns
from redbaron import RedBaron, DefNode, ClassNode
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

def std_code(code): #remove the indent of each line
    lines = code.splitlines()
    char_num = 0
    for char in lines[0]:
        if char == ' ':
            char_num += 1
        else:
            break
    code_std = ''
    for line in lines:
        code_std += line[char_num:] + '\n'
    return (code_std)

def get_abstract_code(code):
    red = RedBaron(code)
    # print(red.help(True))
    defs = red.find_all('def')
    if defs:
        for def_i in defs:
            for decorator in def_i.decorators:
                # print(decorator.value)
                decorator.value = "decorator"
                decorator.call = ""

    classes = red.find_all('class')
    if classes:
        for class_i in classes:
            for decorator in class_i.decorators:
                # print(decorator.value)
                decorator.value = "decorator"
                decorator.call = ""

    string_nodes = red.find_all('string')
    for string_node in string_nodes:
        if string_node.parent is red or isinstance(string_node.parent, DefNode) or isinstance(string_node.parent, ClassNode):
            pass
        else:
            string_node.value = "STRING"

    rstring_nodes = red.find_all('rawstring')
    for rstring_node in rstring_nodes:
        if rstring_node.parent is red or isinstance(rstring_node.parent, DefNode) or isinstance(rstring_node.parent, ClassNode):
            pass
        else:
            rstring_node.value = "STRING"

    InterpolatedStringNodes = red.find_all('InterpolatedStringNode')
    for InterpolatedStringNode in InterpolatedStringNodes:
        if InterpolatedStringNode.parent is red or isinstance(InterpolatedStringNode.parent, DefNode) or isinstance(InterpolatedStringNode.parent, ClassNode):
            pass
        else:
            InterpolatedStringNode.value = "STRING"

    int_nodes = red.find_all('int')
    for int_node in int_nodes:
        int_node.value = "NUMBER"

    int_nodes = red.find_all('float_exponant')
    for int_node in int_nodes:
        int_node.value = "NUMBER"

    float_nodes = red.find_all('float')
    for float_node in float_nodes:
        float_node.value = "NUMBER"


    return(red.dumps())


def get_code_token(code):
    result_str = []
    tmp = re.split('([\[\]{}()| \-,.:;!@#$%^=&*_+\n])', code)
    for word in tmp:
        if word != '\n':
            result_str.append(lemmatizer.lemmatize(word.lower(), pos='v'))
    # print(result_str)
    result_list = []
    for word in result_str:
        if not word.isalnum() and word not in ['', ' ', '@']:
            result_list.append(word)
        elif word != ' ' and word != '':
            result_list.extend(wordninja.split(word))
    return ' '.join(result_list)



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
dict = json.load(open(r'train_data.json', 'r'))
for i in dict:
    # if index < 40000:
    #     index += 1
    #     continue
    try:
        signature = dict[i]['signature']
        # print(get_code_token(dict[i]['source']))
        sig = (get_signature_token(signature))
        if 'desc' in dict[i]:
            desc = (get_desc_token(dict[i]['desc']))
        else:
            desc = ''
        source = dict[i]['source']
        source = std_code(source)
        source = get_abstract_code(source)
        source = (get_code_token(source))
    except Exception:
        continue
    # if index < 2001:
    #     continue
    if index % 500 == 0:
        print(index)
    if index < 40000:
        with open(r'train.token.code', 'a', encoding='utf-8') as wr:
            wr.write(desc +'\n')
        with open(r'train.token.nl', 'a', encoding='utf-8') as wr:
            wr.write(source +'\n')

        with open(r'train.token.sbt', 'a', encoding='utf-8') as wr:
            wr.write(sig +'\n')

    elif index < 43000:
        with open(r'test.token.code', 'a', encoding='utf-8') as wr:
            wr.write(desc +'\n')

        with open(r'test.token.nl', 'a', encoding='utf-8') as wr:
            wr.write(source + '\n')


        with open(r'test.token.sbt', 'a', encoding='utf-8') as wr:
            wr.write(sig +'\n')

    else:
        with open(r'valid.token.code', 'a', encoding='utf-8') as wr:
            wr.write(desc +'\n')

        with open(r'valid.token.nl', 'a', encoding='utf-8') as wr:
            wr.write(source + '\n')


        with open(r'valid.token.sbt', 'a', encoding='utf-8') as wr:
            wr.write(sig +'\n')



    index += 1


print(len(dict))
