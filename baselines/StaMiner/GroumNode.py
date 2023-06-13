from redbaron import RedBaron, IfelseblockNode, AtomtrailersNode, CallNode, NameNode, GetitemNode, NodeList, ForNode,\
                    WhileNode, AssignmentNode, ReturnNode, PrintNode, LineProxyList, ProxyList, CommaProxyList, DotProxyList, \
                    TryNode, CallArgumentNode, ClassNode
import os.path
import ipdb
import ctypes
import sys
sys.setrecursionlimit(8000)

class GroumNode:
    def __init__(self, type, lable, control_lables, call_args):
        self.type = type #type can be 'action' or 'control'
        self.lable = lable
        self.control_lables = []
        for i in control_lables: #Python can only pass mutable variables by reference
            self.control_lables.append(i)
        self.variables = []
        for i in call_args:
            self.variables.append(i)
        # self.next = None

# class Groum:
#     def __init__(self):
#         self.head = None
#
#     def __init__(self, head):
#         self.head = head #type can be 'action' or 'control'

class GroumParser:
    def __init__(self, red, project_path, type):
        self.Groum = {} #store the id of red baron node
        self.GroumNodes = {} #store the GroumNode with their id repectively
        self.control_lable = []#mark the current control node
        self.current_control_bloack = 0
        self.red = red
        self.project_path = project_path
        self.type = type
        self.possible_callargument = set()
        self.involved_control_lables = set()
        self.in_sequence = []
        self.to_be_walked_node = []
        self.already_walked_node = set()
        self.current_sequence = []

    def get_call_with_out_params(self, node): #remove args in callnode
        if isinstance(node, AtomtrailersNode):
            str = ''
            for i in node.value:
                if isinstance(i, CallNode):
                    str += '()'
                elif isinstance(i, NameNode):
                    if str == '':
                        str += i.dumps()
                    else:
                        str += '.' + i.dumps()
                elif isinstance(i, GetitemNode):
                    str += i.dumps()
            return str
        # return node.dumps() # need params

    def get_CallArgument(self, node):
        if isinstance(node, AtomtrailersNode):
            call_args = []
            callnodes = node.find_all('call')
            for callnode in callnodes:
                call_value = callnode.value
                if bool(call_value):
                    for item in call_value:
                        if isinstance(item, CallArgumentNode):
                            namenode = item.value.find_all('name')
                            for name in namenode:
                                if not isinstance(name.parent, AtomtrailersNode): # !!!!!To be optimized!!!!!!
                                    call_args.append(name.value)
                                    self.possible_callargument.add(name.value)
            if isinstance(node.parent, AssignmentNode):
                targets = node.parent.target.find_all('name')
                for target in targets:
                    call_args.append(target.value)
                    self.possible_callargument.add(target.value)
            return call_args


    def add_edge(self, fromnodeid, tonodeid): #judge if fromnode is in Groum
        if isinstance(fromnodeid, list) and isinstance(tonodeid, list):
            for i in fromnodeid:
                if i in self.Groum:
                    for j in tonodeid:
                        if j not in self.Groum[i]:
                            self.Groum[i].append(j)
                else:
                    self.Groum[i] = tonodeid
        elif isinstance(fromnodeid, list):
            for i in fromnodeid:
                if i in self.Groum:
                    if tonodeid not in self.Groum[i]:
                        self.Groum[i].append(tonodeid)
                else:
                    self.Groum[i] = [tonodeid]
        elif isinstance(tonodeid, list):
            for i in tonodeid:
                if fromnodeid in self.Groum:
                    if i not in self.Groum[fromnodeid]:
                        self.Groum[fromnodeid].append(i)
                else:
                    self.Groum[fromnodeid] = [i]
        else:
            if fromnodeid in self.Groum:
                if tonodeid not in self.Groum[fromnodeid]:
                    self.Groum[fromnodeid].append(tonodeid)
            else:
                self.Groum[fromnodeid] = [tonodeid]

    def add_node(self, id, type, lable):
        if not bool(self.GroumNodes):
            self.entry_nodeid = id
        if id not in self.GroumNodes:# add if there not exists the node
            if type is not 'control':
                node = ctypes.cast(id, ctypes.py_object).value
                call_args = self.get_CallArgument(node)
            else:
                call_args = []
            self.GroumNodes[id] = GroumNode(type, lable, self.control_lable, call_args)
            # print(id, self.GroumNodes[id].control_lables)

    # def add_DataDependencies_to(self, node_id):
    #     node = ctypes.cast(node_id, ctypes.py_object).value
    #     call_args = self.get_CallArgument(node)
    #     for call_arg in call_args:
    #         for id , groumnode in self.GroumNodes.items():
    #             if not id == node_id:
    #                 if call_arg in groumnode.variables :
    #                     if node_id in self.Groum:
    #                         if id not in self.Groum[node_id]:
    #                             self.add_edge(id, node_id)
    #                     else:
    #                         self.add_edge(id, node_id)

    def get_common_parentid(self, callnodes): #call_nodes is list
        parentid = set()
        for callnode in callnodes:
            if id(callnode.parent) not in parentid:
                parentid.add(id(callnode.parent))
                self.add_node(id(callnode.parent), 'action', self.get_call_with_out_params(callnode.parent))
                # self.add_DataDependencies_to(id(callnode.parent))
        if len(parentid) > 0:
            parentid = list(parentid)
            return parentid

    def merge_end_node(self, end_node, end_node_cur): #sometimes we have several endnodes(such as IfElseBlocks and TryNode) need to merge it
        if isinstance(end_node, list) and isinstance(end_node_cur, list):
            for i in end_node_cur:
                end_node.append(i)
        elif isinstance(end_node, list):
            end_node.append(end_node_cur)
        elif isinstance(end_node_cur, list):
            end_node_cur.append(end_node)
            end_node = end_node_cur
        else:
            end_node = [end_node, end_node_cur]

        return end_node

    def parsenode(self, node):
        start_node = None
        end_node = None
        if node == self.red:
            for i in node:
                self.parsenode(i)

        elif isinstance(node, IfelseblockNode):
            start_node, end_node = self._parseIfelseblockNode(node)
            return start_node, end_node

        elif isinstance(node, ForNode):
            start_node, end_node = self._parseForNode(node)
            return start_node, end_node

        elif isinstance(node, WhileNode):
            start_node, end_node = self._parseWhileNode(node)
            return start_node, end_node

        elif isinstance(node, TryNode):
            start_node, end_node = self._parseTryNode(node)
            return start_node, end_node

        elif isinstance(node, AssignmentNode): #
            call_order = []
            callnodes = node.find_all('callnode') #!!!!!!!!!!!!!!!!!!Attention callnodes
            if bool(callnodes):
                for callnode in callnodes:
                    atoms = callnode.parent
                    self.add_node(id(atoms), 'action', self.get_call_with_out_params(atoms) )
                    # self.add_DataDependencies_to(id(atoms))
                    parent = atoms.parent
                    while not isinstance(parent, CallNode):
                        if parent is node:
                            break
                        parent = parent.parent
                    next_level_calls = callnode.find_all('call')  # judge if there is still callnode in current call
                    if isinstance(parent, CallNode): #callnode in callnode:
                        if start_node is None and len(next_level_calls) == 1: #there is no callnode in current call
                            start_node = id(atoms)
                        atoms_parent = parent.parent
                        self.add_edge(id(atoms), id(atoms_parent))
                        if id(atoms) not in self.GroumNodes:
                            self.add_node(id(atoms), 'action', self.get_call_with_out_params(atoms))
                            # self.add_DataDependencies_to(id(atoms))
                        if id(atoms_parent) not in self.GroumNodes:
                            self.add_node(id(atoms_parent), 'action', self.get_call_with_out_params(atoms_parent))
                            # self.add_DataDependencies_to(id(atoms_parent))

                    else: #maybe endnode:
                        if start_node is None and len(next_level_calls) == 1:
                            start_node = id(atoms)
                        end_node = id(atoms)
                        self.add_node(id(atoms), 'action', self.get_call_with_out_params(atoms))
                return start_node, end_node#return id of callnode

        elif isinstance(node, AtomtrailersNode):
            callnode = node.find('callnode')
            if callnode is not None:
                atoms = callnode.parent
                self.add_node(id(atoms), 'action', self.get_call_with_out_params(atoms) )
                # self.add_DataDependencies_to(id(atoms))
                return id(atoms), id(atoms)  # return id of callnode

        elif hasattr(node, 'value'):
            if isinstance(node.value, str): #node is NameNode not need to parse
                pass
            elif isinstance(node, PrintNode):
                call_nodes = node.find_all('callnode')
                if bool(call_nodes):
                    parentid = self.get_common_parentid(call_nodes)
                    return parentid, parentid
            elif isinstance(node, ReturnNode):
                call_nodes = node.find_all('callnode')
                if bool(call_nodes):
                    parentid = self.get_common_parentid(call_nodes)
                    return parentid, parentid
            # else:
            elif isinstance(node.value, LineProxyList):
                start_node, end_node = self._parseNodeList(node.value)#we assume that node with attr 'value' is NodeList
                return start_node, end_node

        return None, None

    def _parseNodeList(self, node):
        # print('----------','\n',node.dumps())
        start_node = None
        end_node = None
        latest_id = None

        for i in node:
            start_node_cur, end_node_cur = self.parsenode(i) #start_node_cur from last level
            # print(start_node_cur, end_node_cur)
            if start_node_cur is not None and end_node_cur is not None:
                if start_node is None:
                    start_node = start_node_cur
                    latest_id = end_node_cur
                else:
                    self.add_edge(latest_id, start_node_cur)
                    latest_id = end_node_cur
        if latest_id is not None:
            end_node = latest_id

        return start_node, end_node

    def _parseIfelseblockNode(self, node):
        self.current_control_bloack += 1
        self.control_lable.append(self.current_control_bloack)
        start_node = None
        end_node = None

        ifnode = node.find('if')
        self.add_node(id(ifnode), 'control', 'if' )
        id_of_last_if = id(ifnode)
        call_in_test = ifnode.test.find('callnode') ##!!!!!in case of multiple callnodes
        if call_in_test is not None:
            self.add_node(id(call_in_test.parent), 'action', self.get_call_with_out_params(call_in_test.parent) )
            # self.add_DataDependencies_to(id(call_in_test.parent))
            self.add_edge(id(call_in_test.parent), id(ifnode))
            start_node = id(call_in_test.parent)
        else:
            start_node = id(ifnode)
        start_node_cur, end_node_cur = self.parsenode(ifnode)
        if start_node_cur is not None and end_node_cur is not None:#exists subgroum in ifnode
            self.add_edge(id(ifnode), start_node_cur)
            end_node = end_node_cur
        # print(start_node, end_node)

        elifnodes = node.find_all('elif')
        if bool(elifnodes):
            for elifnode in elifnodes:
                self.add_node(id(elifnode), 'control', 'elif' )
                call_in_test = elifnode.test.find('callnode')
                if call_in_test is not None:
                    self.add_node(id(call_in_test.parent), 'action', self.get_call_with_out_params(call_in_test.parent))
                    # self.add_DataDependencies_to(id(call_in_test.parent))
                    self.add_edge(id_of_last_if, id(call_in_test.parent))
                    self.add_edge(id(call_in_test.parent), id(elifnode))
                else:
                    self.add_edge(id_of_last_if, id(elifnode))
                id_of_last_if = id(elifnode)
                start_node_cur, end_node_cur = self.parsenode(elifnode)
                if start_node_cur is not None and end_node_cur is not None:
                    self.add_edge(id(elifnode), start_node_cur)
                    if end_node is None:#judge if need merge end_node
                        end_node = end_node_cur
                    else:
                        end_node = self.merge_end_node(end_node, end_node_cur)

        elsenode = node.find('else')
        if elsenode is not None:
            start_node_cur, end_node_cur = self.parsenode(elsenode)
            if start_node_cur is not None and end_node_cur is not None:
                self.add_edge(id_of_last_if, start_node_cur)
                if end_node is None:  # judge if need merge end_node
                    end_node = end_node_cur
                else:
                    end_node = self.merge_end_node(end_node, end_node_cur)

        self.control_lable = self.control_lable[0: -1]
        return start_node, end_node

    def _parseForNode(self, node):
        self.current_control_bloack += 1
        self.control_lable.append(self.current_control_bloack)
        start_node = None
        end_node = None
        calls_in_target = node.target.find_all('callnode')
        if bool(calls_in_target):
            start_node = self.get_common_parentid(calls_in_target)
            end_node = start_node
            # print('--------', '\n', len(calls_in_target))
            # for call in calls_in_target:
            #     print(call)
            self.add_edge(start_node, id(node))
        else:
            start_node = id(node)
        self.add_node(id(node), 'control', 'for' )
        for_value = node.value
        # print(for_value)
        start_node_cur, end_node_cur = self._parseNodeList(for_value)
        if start_node_cur is not None and end_node_cur is not None:
            if start_node is None:
                start_node = start_node_cur
            else:
                self.add_edge(id(node), start_node_cur)
            end_node = end_node_cur

        self.control_lable = self.control_lable[0: -1]
        return start_node, end_node

    def _parseWhileNode(self, node):
        self.current_control_bloack += 1
        self.control_lable.append(self.current_control_bloack)
        start_node = None
        end_node = None
        calls_in_test = node.test.find_all('callnode')
        if bool(calls_in_test):
            start_node = self.get_common_parentid(calls_in_test)
            end_node = start_node
            self.add_edge(start_node, id(node))
        else:
            start_node = id(node)
        self.add_node(id(node), 'control', 'while' )
        while_value = node.value
        start_node_cur, end_node_cur = self._parseNodeList(while_value)
        if start_node_cur is not None and end_node_cur is not None:
            if start_node is None:
                start_node = start_node_cur
            else:
                self.add_edge(id(node), start_node_cur)
            end_node = end_node_cur

        self.control_lable = self.control_lable[0: -1]
        return start_node, end_node

    def _parseTryNode(self, node):
        self.current_control_bloack += 1
        self.control_lable.append(self.current_control_bloack)
        start_node = None
        end_node = None
        try_value = node.value
        self.add_node(id(node), 'control', 'try')
        start_node = id(node)
        start_node_cur, end_node_cur = self._parseNodeList(try_value)
        if start_node_cur is not None and end_node_cur is not None:
            self.add_edge(start_node, start_node_cur)
            end_node = end_node_cur
        excepts = node.find('except')
        except_value = excepts.value
        start_node_cur, end_node_cur = self._parseNodeList(except_value)
        if start_node_cur is not None and end_node_cur is not None:
            self.add_edge(start_node, start_node_cur)
            if end_node is None: # judge if we need to merge end nodes
                end_node = end_node_cur
            else:
                self.merge_end_node(end_node, end_node_cur)

        self.control_lable = self.control_lable[0: -1]
        return start_node, end_node

    def MethodDeclarationParser(self, MethodDeclarationNodes):
        for MethodDeclarationNode in MethodDeclarationNodes:
            if not os.path.isdir(self.project_path):
                os.mkdir(self.project_path)
            if isinstance(MethodDeclarationNode.parent, ClassNode):
                dst_dir = self.project_path + r'/%s' % MethodDeclarationNode.parent.name
                if not os.path.isdir(dst_dir):
                    os.mkdir(dst_dir)
                writer = open(dst_dir + r'/%s.txt' % MethodDeclarationNode.name, 'a')
            else:
                if self.type == 'tf':
                    writer = open(self.project_path + r'/%s_tf.txt' % MethodDeclarationNode.name, 'a')
                else:
                    writer = open(self.project_path + r'/%s_torch.txt' % MethodDeclarationNode.name, 'a')
            # print(MethodDeclarationNode.name, '------')
            self.Groum = {}  # store the id of red baron node
            self.GroumNodes = {}  # store the GroumNode with their id repectively
            self.control_lable = []  # mark the current control node
            self.current_control_bloack = 0
            self.red = red
            self.possible_callargument = set()
            self.involved_control_lables = set()
            self.in_sequence = []
            self.to_be_walked_node = []
            self.already_walked_node = set()
            self.current_sequence = []
            # print('\n', '--------------------------', MethodDeclarationNode.name)
            self.parsenode(MethodDeclarationNode)
            for key, value in self.Groum.items():
                # print()
                # print(key, '   ', self.GroumNodes[key].lable, '   ', self.GroumNodes[key].control_lables, \
                #       '   ', self.GroumNodes[key].variables,'\n', '------')
                for i in value:
                    if isinstance(i, list):
                        for j in i:
                            pass
                            # print(j, '   ', self.GroumNodes[j].lable, '  ', self.GroumNodes[j].control_lables, \
                            #       '   ', self.GroumNodes[j].variables)
                    else:
                        pass
                        # print(i, '   ', self.GroumNodes[i].lable, '  ', self.GroumNodes[i].control_lables, \
                        #       '   ', self.GroumNodes[i].variables)
            self.print_sequence(self.possible_callargument)
            # print(len(self.GroumNodes))
            for apiid in (self.sort_sequence(self.current_sequence)):
                # writer.write(self.GroumNodes[apiid].lable + ' ,')
                # print(self.GroumNodes[apiid].lable)
                self.parse_self_part(apiid, writer)
            writer.close()
            # print(self.possible_callargument)

    def parse_self_part(self, apiid, writer): #inline the 'self.'
        need_to_parse = False
        lable = self.GroumNodes[apiid].lable
        if lable.find('self.') is not -1:
            tmp = lable.strip('()')
            api = ctypes.cast(apiid, ctypes.py_object).value
            parent = api.parent
            while not isinstance(parent, ClassNode):
                if parent is self.red:
                    break
                parent = parent.parent
            if parent is not self.red:
                assign = parent.find('assign', target = lambda x: x.dumps() == tmp)
                if assign is not None:
                    layer = assign.value
                    layer_str = layer.dumps()
                    layername = layer_str.split('(')[0]
                    layerclass = self.red.find('class', name = layername)
                    if layerclass is not None:
                        tf_call_method = layerclass.find('defnode', name = 'call')
                        if tf_call_method is not None:
                            need_to_parse = True
        if need_to_parse:
            # print('lable:------', tf_call_method.name, 'class:-------', tf_call_method.parent.name)
            a = self.Groum
            b = self.GroumNodes
            c = self.control_lable
            d = self.current_control_bloack
            e = self.possible_callargument
            f = self.involved_control_lables
            g = self.in_sequence
            h = self.to_be_walked_node
            i = self.already_walked_node
            j = self.current_sequence
            k = self.red

            self.MethodDeclarationParser([tf_call_method])

            self.Groum = a  # store the id of red baron node
            self.GroumNodes = b  # store the GroumNode with their id repectively
            self.control_lable = c  # mark the current control node
            self.current_control_bloack = d
            self.possible_callargument = e
            self.involved_control_lables = f
            self.in_sequence = g
            self.to_be_walked_node = h
            self.already_walked_node = i
            self.current_sequence = j
            self.red = k
            # print('DONE')
        else:
            # if self.GroumNodes[apiid].type == 'action':
            writer.write(self.GroumNodes[apiid].lable + '\n')
            pass




    def print_sequence(self, args): #obtain the correct sequece but not 'print'
        try:
            self.get_involved_control_lables(self.entry_nodeid, args)
        except BaseException:
            self.involved_control_lables = set()
        for key, value in self.Groum.items():
            if self.has_data_dependency(args, self.GroumNodes[key].variables) and key not in self.in_sequence:
                # print(key, '   ', self.GroumNodes[key].lable, '   ', self.GroumNodes[key].control_lables, \
                #       '   ', self.GroumNodes[key].variables)
                self.current_sequence.append(key)
                self.in_sequence.append(key)
            if self.GroumNodes[key].type == 'control' and key not in self.in_sequence:
                to_be_remained = True
                for lbs in self.GroumNodes[key].control_lables:
                    if lbs not in self.involved_control_lables:
                        to_be_remained = False
                        break
                if to_be_remained:
                    # print(key, self.GroumNodes[key].lable, ',', self.GroumNodes[key].control_lables)
                    self.current_sequence.append(key)
                    self.in_sequence.append(key)
            for i in value:
                if isinstance(i, list):
                    for j in i:
                        if (self.has_data_dependency(args, self.GroumNodes[j].variables) and j not in self.in_sequence):
                            # print(j, '   ', self.GroumNodes[j].lable, '  ', self.GroumNodes[j].control_lables, \
                            #       '   ', self.GroumNodes[j].variables)
                            self.current_sequence.append(j)
                            self.in_sequence.append(j)
                        if self.GroumNodes[j].type == 'control' and j not in self.in_sequence:
                            to_be_remained = True
                            for lbs in self.GroumNodes[j].control_lables:
                                if lbs not in self.involved_control_lables:
                                    to_be_remained = False
                                    break
                            if to_be_remained:
                                # print(j, self.GroumNodes[j].lable, ',', self.GroumNodes[j].control_lables)
                                self.current_sequence.append(j)
                                self.in_sequence.append(j)
                else:
                    if self.has_data_dependency(args, self.GroumNodes[i].variables) and i not in self.in_sequence:
                        # print(i, '   ', self.GroumNodes[i].lable, '  ', self.GroumNodes[i].control_lables, \
                        #       '   ', self.GroumNodes[i].variables)
                        self.current_sequence.append(i)
                        self.in_sequence.append(i)
                    if self.GroumNodes[i].type == 'control' and i not in self.in_sequence:
                        to_be_remained = True
                        for lbs in self.GroumNodes[i].control_lables:
                            if lbs not in self.involved_control_lables:
                                to_be_remained = False
                                break
                        if to_be_remained:
                            # print(i, self.GroumNodes[i].lable, ',', self.GroumNodes[i].control_lables)
                            self.current_sequence.append(i)
                            self.in_sequence.append(i)

    def sort_sequence(self, sequence):
        new_seq = sequence
        i = 0
        while i < len(sequence):
            if self.GroumNodes[sequence[i]].type == "control":
                control_lables = self.GroumNodes[sequence[i]].control_lables
                j = 0
                while j <= i:
                    if self.GroumNodes[sequence[j]].type == "action" and\
                        self.GroumNodes[sequence[j]].control_lables == control_lables:
                        control_node_id = new_seq.pop(i)
                        new_seq.insert(j, control_node_id)
                        break
                    j += 1
            i += 1
        return new_seq

    def has_data_dependency(self, args, node_variables):
        has = False
        for node_variable in node_variables:
            if node_variable in args:
                has = True
                break
        return has

    def get_involved_control_lables(self, root, args): #to determine which control nodes are involved
        if root in self.Groum:
            if self.has_data_dependency(args, self.GroumNodes[root].variables):
                for lbs in self.GroumNodes[root].control_lables:
                    self.involved_control_lables.add(lbs)
            children = self.Groum[root]
            for child in children:
                if isinstance(child, list):
                    for i in child:
                        self.get_involved_control_lables(i, args)
                else:
                    self.get_involved_control_lables(child, args)

    def walk_Groum(self):
        if bool(self.to_be_walked_node):
            root = self.to_be_walked_node[0]
            if root in self.GroumNodes and root not in self.already_walked_node:
                self.already_walked_node.add(root)
                # if 'array' in self.GroumNodes[root].variables :
                #     print(root, ',', self.GroumNodes[root].lable, ',')
                # if self.GroumNodes[root].type == 'control':
                #     for lbs in self.GroumNodes[root].control_lables:
                #         if lbs in self.involved_control_lables:
                #             print(self.GroumNodes[root].lable, ',')
                #             break
                if root in self.Groum:
                    children = self.Groum[root]
                    # print(children)
                    for child in children:
                        if isinstance(child, list):
                            for i in child:
                                self.to_be_walked_node.append(i)
                                if 'array' in self.GroumNodes[i].variables:
                                    print(i, ',', self.GroumNodes[i].lable, ',')
                        else:
                            self.to_be_walked_node.append(child)
                            if 'array' in self.GroumNodes[child].variables:
                                print(child, ',', self.GroumNodes[child].lable, ',')
            self.to_be_walked_node.pop(0)
            self.walk_Groum()



tf_path = r"the file need to be parsed"
dst_path_tf = r'the output dir'


with open(tf_path, "r") as reader:
    src = reader.read()
red_tf = RedBaron(src)


groumparser = GroumParser(red_tf, dst_path_tf, 'tf')
MethodDeclarationNodes = red_tf.find_all('defnode')
groumparser.MethodDeclarationParser(MethodDeclarationNodes)
