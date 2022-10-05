import random
import re
import time
import functools
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.semparse import util
from utils.sparql_cache import SparqlCache


path = str(Path(__file__).parent.absolute())


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds for {func.__name__}")
        return value
    return wrapper_timer


def get_vocab(dataset: str):
    if dataset == "grail":
        with open(path + '/vocab_files/grailqa.json') as f:
            data = json.load(f)
        return set(data["relations"]), set(data["classes"]), set(data["attributes"])
    elif dataset == "graphq":
        with open(path + '/vocab_files/gq1.json') as f:
            data = json.load(f)
        return set(data["relations"]), set(data["classes"]), set(data["attributes"])
    elif dataset == "webq":
        # with open(path + '/vocab_files/webq.json') as f:
        with open(path + '/vocab_files/webq_full.json') as f:
            data = json.load(f)
        return set(data["relations"]), set(data["classes"]), set(data["attributes"]), set(data["tc_attributes"]), set(
            data["cons_attributes"]), data["cons_ids"]
    elif dataset == "cwq":
        pass


def get_ontology(dataset: str):
    class_hierarchy = defaultdict(lambda: [])
    class_out_edges = defaultdict(lambda: set())
    class_in_edges = defaultdict(lambda: set())
    relation_domain = {}
    relation_range = {}
    date_attributes = set()
    numerical_attributes = set()
    if dataset == "grail":
        fb_type_file = path + "/../ontology/commons/fb_types"
        fb_roles_file = path + "/../ontology/commons/fb_roles"
    elif dataset == "graphq":
        fb_type_file = path + "/../ontology/fb_types"
        fb_roles_file = path + "/../ontology/fb_roles"

    else:  # webq does not need these information
        return class_out_edges, class_in_edges, relation_domain, relation_range, date_attributes, numerical_attributes

    with open(fb_type_file) as f:
        for line in f:
            fields = line.split()
            if fields[2] != "common.topic":
                class_hierarchy[fields[0]].append(fields[2])

    with open(fb_roles_file) as f:
        for line in f:
            fields = line.split()
            relation_domain[fields[1]] = fields[0]
            relation_range[fields[1]] = fields[2]

            class_out_edges[fields[0]].add(fields[1])
            class_in_edges[fields[2]].add(fields[1])

            if fields[2] in ['type.int', 'type.float']:
                numerical_attributes.add(fields[1])
            elif fields[2] == 'type.datetime':
                date_attributes.add(fields[1])

    for c in class_hierarchy:
        for c_p in class_hierarchy[c]:
            class_out_edges[c].update(class_out_edges[c_p])
            class_in_edges[c].update(class_in_edges[c_p])

    return class_out_edges, class_in_edges, relation_domain, relation_range, date_attributes, numerical_attributes


class KBEngine:
    def __init__(self, dataset='grail', MAX_VARIABLES_NUM=20):
        self._dataset = dataset
        if dataset in ["graphq", "grail"]:
            self._relations, self._classes, self._attributes = get_vocab(dataset)
        elif dataset == "webq":
            self._relations, self._classes, self._attributes, self._tc_attributes, self._cons_attributes, self._cons_ids = get_vocab(
                dataset)

        if dataset == "grail":
            with open('ontology/domain_dict', 'r') as f:
                self._domain_dict = json.load(f)
            with open('ontology/domain_info', 'r') as f:
                self._domain_info = json.load(f)
        self._class_out, self._class_in, self._relation_d, self._relation_r, self._date_attributes, \
        self._numerical_attributes = get_ontology(dataset)
        self._date_attributes = self._date_attributes.intersection(self._attributes)
        self._numerical_attributes = self._numerical_attributes.intersection(self._attributes)
        self._cache = SparqlCache(dataset)
        self.training = False
        self.max_variables_num = MAX_VARIABLES_NUM

    def get_vocab(self):
        return self._relations, self._classes, self._attributes

    def set_training(self, training):  # call it at the beginning of each forward pass
        self.training = training

    def process_value(self, value):
        data_type = value.split("^^")[1].split("#")[1]
        if data_type not in ['integer', 'float', 'double', 'dateTime']:
            value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
            # value = value.split("^^")[0] + '-08:00^^' + value.split("^^")[1]
        else:
            value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

        return value

    def get_relations_for_variables(self, entities, reverse=False, add_noise=False):
        '''
        The most straightforward way is obviously get those relations using SPARQL query, but I am not sure about
        the efficiency of doing this.
        Also, for debug purpose, we can also just simply return all the relations in Freebase to make sure the whole
        flow works.
        :param entities: A set of entities
        :param reverse: True indicates outgoing relations, while False indicates ingoing relations
        :return: All adjacent relations of those entities
        '''

        # if TC:
        #     tc_relations = set()
        #     for r in self._relations:
        #         if r.__contains__(".from"):
        #             tc_relations.add(r)
        #     return tc_relations

        # print("get relations for: {} entities".format(len(entities)))
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:20]:
            try:
                if reverse:
                    rtn.update(self._cache.get_out_relations(entity).intersection(self._relations))
                else:
                    rtn.update(self._cache.get_in_relations(entity).intersection(self._relations))
            except Exception:
                # print("entity:", entity)
                pass
        # print(entities)
        # print("done getting relations")

        if self.training and add_noise:
            if not self._dataset == 'grail':
                rtn.update(random.sample(self._relations, 100))
            elif len(self._domains) > 0:
                if random.random() > 0.5:
                    vocab = set()
                    for d in self._domains:
                        vocab.update(self._domain_dict[d])
                    # rtn = rtn.intersection(vocab)
                    if len(vocab) > 100:
                        rtn.update(random.sample(vocab, 100))
                    else:
                        rtn.update(vocab)

        return rtn

    def get_relations_for_class(self, class_name, reverse=False, add_noise=False):
        if reverse:
            return self._class_out[class_name].intersection(self._relations)
        else:
            return self._class_in[class_name].intersection(self._relations)

    def get_attributes_for_variables(self, entities, add_noise=False):
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:20]:
            try:
                rtn.update(self._cache.get_out_relations(entity).intersection(self._attributes))
            except Exception:
                # print("entity:", entity)
                pass
        # print(entities)
        # print("done getting relations")

        if self.training and add_noise:
            if len(self._attributes) > 100:
                rtn.update(random.sample(self._attributes, 100))
            else:
                rtn.update(self._attributes)

        return rtn

    def get_tc_attributes_for_variables(self, entities, add_noise=False):
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:20]:
            try:
                rtn.update(self._cache.get_out_relations(entity).intersection(self._tc_attributes))
            except Exception:
                # print("entity:", entity)
                pass

        if self.training and add_noise:
            if len(self._tc_attributes) > 100:
                rtn.update(random.sample(self._tc_attributes, 100))
            else:
                rtn.update(self._tc_attributes)

        return rtn

    def get_cons_attributes_for_variables(self, entities, add_noise=False):
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:20]:
            try:
                rtn.update(self._cache.get_out_relations(entity).intersection(self._cons_attributes))
            except Exception:
                # print("entity:", entity)
                pass

        if self.training and add_noise:
            if len(self._cons_attributes) > 100:
                rtn.update(random.sample(self._cons_attributes, 100))
            else:
                rtn.update(self._cons_attributes)

        return rtn

    def get_attributes_for_value(self, value, add_noise=False, use_ontology=True):
        rtn = set()

        if use_ontology:
            if value.__contains__("#float") or value.__contains__("#integer") or value.__contains__("#double"):
                rtn.update(self._numerical_attributes)
            else:
                rtn.update(self._date_attributes)
        else:  # retrieve based on KB facts
            data_type = value.split("#")[1]
            if data_type not in ['integer', 'float', 'double', 'dateTime']:
                value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
            else:
                value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

            rtn.update(self._cache.get_in_attributes(value).intersection(self._attributes))

        if self.training and add_noise:
            if len(self._attributes) > 100:
                rtn.update(random.sample(self._attributes, 100))
            else:
                rtn.update(self._attributes)

        return rtn

    def get_attributes_for_class(self, class_name, add_noise=False):
        return self._class_out[class_name].intersection(self._attributes)

    def is_intersectant(self, derivation1, derivation2):
        return self._cache.is_intersectant(derivation1, derivation2)

    def get_reachable_classes(self, derivations, answer_types):
        reachable_classes = set()
        for a in answer_types:
            flag = True
            for d in derivations:
                if not self._cache.is_reachable(d, a):
                    flag = False
                    break
            if flag:
                reachable_classes.add(a)

        return reachable_classes

    def get_classes_for_variables(self, entities, add_noise=False):
        # print("get classes for: {} entities".format(len(entities)))
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:20]:
            rtn.update(set(self._cache.get_types(entity)).intersection(self._classes))

        if self.training and add_noise:
            if not self._dataset == "grail":
                if len(self._classes) > 100:
                    rtn.update(random.sample(self._classes, 100))
                else:
                    rtn.update(self._classes)
            elif len(self._domains) > 0:
                if random.random() > 0.5:
                    vocab = set()
                    for d in self._domains:
                        vocab.update(self._domain_dict[d])
                    # rtn = rtn.intersection(vocab)
                    if len(vocab) > 100:
                        rtn.update(random.sample(vocab, 100))
                    else:
                        rtn.update(vocab)

        return rtn

        # return classes

    def get_constraints_for_variables(self, entities, cons_attribute):
        rtn = set()
        # TODO: remove this constraint, this is only for debugging.
        for entity in list(entities)[:20]:
            rtn.update(set(self._cache.get_out_entities(entity, cons_attribute)).intersection(self._cons_ids))

        return rtn

    def execute_AND(self, arg1, arg2):
        if not isinstance(arg2, set):
            rtn = set()
            # TODO: this is only for debug
            for entity in list(arg1)[:20]:
                if arg2 in self._cache.get_types(entity):
                    rtn.add(entity)
            return rtn
        else:
            return arg1.intersection(arg2)

    def execute_JOIN(self, arg1, arg2):
        # print("execute JOIN for: {} entities".format(len(arg1)))
        rtn = set()
        if isinstance(arg1, str):
            value = arg1
            data_type = value.split("^^")[1].split("#")[1]
            if data_type not in ['integer', 'float', 'double', 'dateTime']:
                value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
                # value = value.split("^^")[0] + '-08:00^^' + value.split("^^")[1]
            else:
                value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

            rtn.update(self._cache.get_in_entities_for_literal(value, arg2))
        else:
            if arg2[-4:] == '_inv':
                # TODO: this is only for debug
                for entity in list(arg1)[:20]:
                    # print(entity, arg2[1])
                    rtn.update(self._cache.get_out_entities(entity, arg2[:-4]))
            else:
                # TODO: this is only for debug
                for entity in list(arg1)[:20]:
                    # print(arg2, entity)
                    rtn.update(self._cache.get_in_entities(entity, arg2))
        # print("done executing JOIN")
        return rtn

    def execute_TC(self, arg1, arg2, arg3):
        # TODO: apply time constraint (not urgent)
        return arg1

    def execute_Comparative(self, arg1, arg2, comparator):
        assert isinstance(arg1, str)  # it must be a value instead of a set of entities
        value = arg1
        if comparator == 'le':
            comp = '<='
        elif comparator == 'lt':
            comp = '<'
        elif comparator == 'ge':
            comp = '>='
        elif comparator == 'gt':
            comp = '>'

        data_type = value.split("^^")[1].split("#")[1]
        if data_type not in ['integer', 'float', 'double', 'dateTime']:
            value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
            # value = value.split("^^")[0] + '-08:00^^' + value.split("^^")[1]
        else:
            value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

        rtn = set()
        rtn.update(self._cache.get_entities_cmp(value, arg2, comp))

        return rtn

    # @timer
    def get_admissible_actions(self, predictions, variables, num_topics, arg_mode=False, arg_variables=set(),
                               arg_class=None,
                               add_noise=False,
                               domains=None,
                               answer_types=None,
                               derivations=None,
                               initial_map=None):
        """
        This helps to get the admissible actions of a decoding step given predictions from previous 2 steps
        :param predictions:
        :param variables:
        :param num_topics: number of topic entities (values)
        :param arg_mode: indicate now it's inside a superlative function
        :return:
        """

        if domains is not None and len(domains) > 0:
            assert self._dataset == 'grail'
            self._domains = domains
        else:
            self._domains = []

        admissible_constants = []
        admissible_variables = []
        # computing a score for all vocab items is intractable, so we use a small set of items to represent the
        # entire vocab
        representative_vocab_items = []

        if len(predictions) == 1:
            two_tokens = [None, predictions[0]]
        else:
            two_tokens = predictions[-2:]

        token = two_tokens[1]
        if token == START_SYMBOL:
            admissible_constants.append('(')
        elif token in [END_SYMBOL, "@@UNKNOWN@@", "@@PADDING@@"]:
            admissible_constants.append(END_SYMBOL)
        elif token == '(':
            if two_tokens[0][0] != '#' and two_tokens[0] not in self._classes:
                if two_tokens[0] != START_SYMBOL:
                    if self._dataset in ['graphq', 'grail']:
                        admissible_constants.extend(
                            ['AND', 'JOIN', 'ARGMAX', 'ARGMIN', 'COUNT', 'lt', 'le', 'gt', 'ge'])
                    elif self._dataset == "webq":
                        admissible_constants.extend(['AND', 'JOIN', 'TC', 'CONS', 'ARGMAX', 'ARGMIN'])
                    elif self._dataset == "cwq":
                        admissible_constants.extend(['AND', 'JOIN', 'TC', 'CONS', 'ARGMAX', 'ARGMIN', 'lt', 'gt'])

                else:  # The first function
                    if self._dataset in ['graphq', 'grail']:
                        if len(variables) == 0:
                            admissible_constants.extend(['ARGMAX', 'ARGMIN'])
                        #  for grailqa, it could be (ARGMAX class relation)
                        else:
                            admissible_constants.extend(
                                ['JOIN', 'lt', 'le', 'gt', 'ge', 'ARGMAX', 'ARGMIN'])
                    elif self._dataset == "webq":
                        admissible_constants.extend(['JOIN'])
                    elif self._dataset == "cwq":
                        admissible_constants.extend(['JOIN', 'lt', 'gt'])

        elif token == ')':
            if predictions[-3] == 'COUNT':
                admissible_constants.append(END_SYMBOL)  # COUNT function is never nested
            elif arg_mode and two_tokens[0] in self._attributes:
                admissible_constants.append(END_SYMBOL)  # ARG function is never nested
            else:
                if len(variables) < self.max_variables_num:
                    admissible_constants.extend([END_SYMBOL, '('])
                else:
                    admissible_constants.append(END_SYMBOL)  # force to stop
        elif token in ['AND', 'COUNT', 'TC', 'CONS']:
            for i in range(len(variables)):
                if isinstance(variables[i], set):
                    if i >= num_topics:  # can only be applied to intermediate executions
                        admissible_variables.append(i)
        elif token == 'JOIN':
            if self._dataset in ["graphq", "grail"]:
                for i in range(len(variables)):  # JOIN accepts both entities and literals
                    admissible_variables.append(i)
            elif self._dataset == "webq":
                for i in range(len(variables)):
                    if isinstance(variables[i], str) and not re.match("[\d]{4}", variables[i]):
                        admissible_variables.append(i)
                    elif isinstance(variables[i], set):
                        admissible_variables.append(i)

        elif token in ['ARGMAX', 'ARGMIN']:
            for i in range(len(variables)):
                if isinstance(variables[i], set):
                    if i >= num_topics:
                        admissible_variables.append(i)
            if len(admissible_variables) == 0 and self._dataset == "grail":
                if self.training:
                    admissible_constants.extend(self._classes)
                else:
                    admissible_constants.extend(answer_types)
            elif len(admissible_variables) == 0 and self._dataset == "graphq":
                for c in self._classes:
                    if c[:7] != 'common.' and c[:5] != 'type.' and c[:3] != 'kg.' and c[:5] != 'user.' \
                            and c[:5] != 'base.':
                        admissible_constants.append(c)
            # else the first argument should not be a class
        elif token in ['le', 'lt', 'ge', 'gt']:
            for i in range(len(variables)):
                if isinstance(variables[i], str):
                    admissible_variables.append(i)
        elif token == 'NOW':
            admissible_constants.append(')')
        elif token[0] == '#':
            if two_tokens[0] == 'AND':
                if self._dataset in ['graphq', 'grail']:
                    if self.training or self._dataset == 'graphq':
                        admissible_constants.extend(
                            self.get_classes_for_variables(variables[int(token[1:])], add_noise=add_noise))

                # admissible_constants.extend(
                #     self.get_classes_for_variables(variables[int(token[1:])], add_noise=add_noise).intersection(
                #         set(answer_types)))
                    else:
                        try:
                            derivations_list = []
                            for k in derivations[int(token[1:])]:
                                if len(derivations[int(token[1:])][k]) == 0:  # no derivation info
                                    continue
                                start = variables[k]
                                if isinstance(start, set):
                                    derivations_list.append(
                                        [':' + list(start)[0], derivations[int(token[1:])][k][0]])
                                else:
                                    value = self.process_value(start)
                                    if len(derivations[int(token[1:])][k]) == 2:
                                        derivations_list.append([value, derivations[int(token[1:])][k][0],
                                                                 derivations[int(token[1:])][k][1]])
                                    else:
                                        derivations_list.append([value, derivations[int(token[1:])][k][0]])

                                admissible_constants.extend(
                                    self.get_reachable_classes(derivations_list, answer_types))

                        except Exception:
                            admissible_constants.extend(
                                self.get_classes_for_variables(variables[int(token[1:])],
                                                               add_noise=add_noise).intersection(set(answer_types)))

                elif self._dataset == 'webq':
                    admissible_constants.extend(
                        self.get_classes_for_variables(variables[int(token[1:])], add_noise=add_noise))

                admissible_v = []
                if isinstance(variables[int(token[1:])], set):
                    for i in range(len(variables)):
                        if i != int(token[1:]):
                            if isinstance(variables[i], set):
                                if i >= num_topics:
                                    # if len(variables[i].intersection(variables[int(token[1:])])) > 0:
                                    #  currently only support path sub-queries intersection
                                    try:
                                        assert len(derivations[i]) == 1
                                        derivation_1 = []
                                        for k in derivations[i]:
                                            start = variables[k]
                                            if isinstance(start, set):
                                                derivation_1.append(':' + list(start)[0])
                                            else:
                                                value = self.process_value(start)
                                                derivation_1.append(value)
                                            derivation_1.append(derivations[i][k][0])
                                            if len(derivations[i][k]) == 2:
                                                derivation_1.append(derivations[i][k][1])
                                        assert len(derivations[int(token[1:])]) == 1
                                        derivation_2 = []
                                        for k in derivations[int(token[1:])]:
                                            start = variables[k]
                                            if isinstance(start, set):
                                                derivation_2.append(':' + list(start)[0])
                                            else:
                                                value = self.process_value(start)
                                                derivation_2.append(value)
                                            derivation_2.append(derivations[int(token[1:])][k][0])
                                            if len(derivations[int(token[1:])][k]) == 2:
                                                derivation_2.append(derivations[int(token[1:])][k][1])
                                        if self.is_intersectant(derivation_1, derivation_2):
                                            admissible_v.append(i)
                                    except Exception:
                                        if len(variables[i].intersection(variables[int(token[1:])])) > 0:
                                            admissible_v.append(i)
                admissible_variables.extend(admissible_v)
            elif two_tokens[0] in ['JOIN']:
                if isinstance(variables[int(token[1:])], str):  # literal
                    if self._dataset in ["graphq", "grail"]:
                        admissible_constants.extend(
                            self.get_attributes_for_value(variables[int(token[1:])], add_noise=add_noise,
                                                          use_ontology=False)
                        )
                    elif self._dataset == "webq":
                        admissible_constants.append("sports.sports_team_roster.number")
                else:
                    admissible_constants.extend(
                        self.get_relations_for_variables(variables[int(token[1:])], add_noise=add_noise))
                    admissible_constants.extend(map(lambda x: x + '_inv',
                                                    self.get_relations_for_variables(variables[int(token[1:])],
                                                                                     reverse=True,
                                                                                     add_noise=add_noise)))
            elif two_tokens[0] in ['le', 'lt', 'ge', 'gt']:
                assert isinstance(variables[int(token[1:])], str)
                admissible_constants.extend(
                    self.get_attributes_for_value(variables[int(token[1:])], add_noise=add_noise)
                )
            elif two_tokens[0] in ['ARGMAX', 'ARGMIN']:
                admissible_constants.extend(
                    self.get_relations_for_variables(variables[int(token[1:])], reverse=True, add_noise=add_noise))
                admissible_constants.extend(map(lambda x: x + '_inv',
                                                self.get_relations_for_variables(variables[int(token[1:])],
                                                                                 add_noise=add_noise)))
                admissible_constants.extend(
                    self.get_attributes_for_variables(variables[int(token[1:])], add_noise=add_noise))
            elif two_tokens[0] == 'COUNT':
                admissible_constants.extend([')'])
            elif two_tokens[0] == 'TC':
                assert self._dataset in ["webq", "cwq"]
                admissible_constants.extend(
                    self.get_tc_attributes_for_variables(variables[int(token[1:])], add_noise=add_noise))
            elif two_tokens[0] == "CONS":
                assert self._dataset in ["webq", "cwq"]
                admissible_constants.extend(
                    self.get_cons_attributes_for_variables(variables[int(token[1:])], add_noise=add_noise))
            elif two_tokens[0][0] == '#':  # (AND #1 #2)
                admissible_constants.extend([')'])
            else:
                if self._dataset in ["graphq", "grail"]:
                    print("Unexpected:", two_tokens[0])
                elif self._dataset == "webq":
                    assert isinstance(variables[int(token[1:])], str)
                    assert two_tokens[0] in self._tc_attributes
                    admissible_constants.append(')')
        elif self._dataset in ["graphq", "grail"] and token in self._attributes:
            admissible_constants.append(')')
        elif self._dataset in ["webq", "cwq"] and token in self._tc_attributes and len(predictions) > 2 and predictions[
            -3] == 'TC':
            # Actually len(predictions) should always be at least 3 here, but at the early stage of beam search,
            # it's possible to include some illegal actions to fill the beam size
            for i in range(len(variables)):
                if isinstance(variables[i], str):
                    if re.match("[\d]{4}", variables[i]):  # year
                        admissible_variables.append(i)
            if len(admissible_variables) == 0:
                admissible_constants.append('NOW')
        elif self._dataset in ["webq", "cwq"] and token in self._cons_attributes and len(predictions) > 2 and \
                predictions[-3] == 'CONS':
            admissible_constants.extend(self.get_constraints_for_variables(variables[int(predictions[-2][1:])], token))
            # admissible_constants.extend(self._cons_ids)
        elif self._dataset in ["webq", "cwq"] and token in self._cons_ids:
            admissible_constants.append(')')
        elif token in self._relations and token in self._attributes:
            # for cwq and webq, it's possible to have overlap between attributes and relations
            admissible_constants.append(')')
        elif token in self._relations or token.replace("_inv", '') in self._relations:
            if arg_mode:
                if arg_class is not None:
                    admissible_constants.extend(self.get_attributes_for_class(arg_class, add_noise=add_noise))

                    admissible_constants.extend(
                        self.get_relations_for_class(arg_class, reverse=True, add_noise=add_noise))
                    admissible_constants.extend(
                        map(lambda x: x + '_inv',
                            self.get_relations_for_class(arg_class, reverse=False, add_noise=add_noise))
                    )
                else:
                    admissible_constants.extend(
                        self.get_attributes_for_variables(arg_variables, add_noise=add_noise))

                    admissible_constants.extend(
                        self.get_relations_for_variables(arg_variables, reverse=True, add_noise=add_noise))

                    admissible_constants.extend(map(lambda x: x + '_inv',
                                                    self.get_relations_for_variables(arg_variables,
                                                                                     add_noise=add_noise)))
            else:
                admissible_constants.extend([')'])
        elif self._dataset in ["webq", 'cwq'] and token in self._attributes:
            admissible_constants.append(')')
        elif token in self._classes:
            # I made a change to datareader, now variable is forced to be precede class, so the following
            # doesn't make sense any more.
            # admissible_variables.extend([i for i in range(len(variables))])
            if two_tokens[0] not in ['ARGMAX', 'ARGMIN']:
                admissible_constants.extend([')'])
            else:  # This is not gonna happen for webq or cwq
                assert self._dataset in ["graphq", "grail"]
                admissible_constants.extend(self.get_relations_for_class(token, reverse=True, add_noise=add_noise))
                admissible_constants.extend(self.get_attributes_for_class(token, add_noise=add_noise))

        if self.training:
            if len(admissible_constants) == 0 and len(admissible_variables) == 0:
                admissible_constants.append(END_SYMBOL)  # In this way we can get infinite loss we want

        for p in predictions:
            if p[0] == "#":
                if int(p[1:]) in admissible_variables:
                    admissible_variables.remove(int(p[1:]))  # TODO: for webq, this may not be true
                if int(p[1:]) < num_topics:   #  remove variables of the same mention
                    mention = initial_map[int(p[1:])][1]
                    to_remove = []
                    for v in admissible_variables:
                        if v < num_topics and initial_map[v][1] == mention:
                            to_remove.append(v)
                    for v in to_remove:
                        admissible_variables.remove(v)


        if len(admissible_variables) > 0:
            admissible_constants = []


        return admissible_constants, admissible_variables, representative_vocab_items