import re
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict
import random
import logging
import json
from pathlib import Path
from nltk import word_tokenize

from overrides import overrides
from allennlp.semparse import util
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from utils.sparql_executer import get_notable_type

logger = logging.getLogger(__name__)
path = str(Path(__file__).parent.absolute())


@DatasetReader.register("arcane_seq2seq")
class Arcane_DatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``ComposedSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens : ``TextField`` and
        target_tokens : ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    # Parameters

    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``SpacyTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    source_add_end_token : bool, (optional, default=True)
        Whether or not to add `END_SYMBOL` to the end of the source sequence.
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """

    def __init__(
            self,
            using_plm: bool = False,
            source_tokenizer: Tokenizer = None,
            source_indexer: TokenIndexer = None,
            target_tokenizer: Tokenizer = None,
            source_token_indexers: Dict[str, TokenIndexer] = None,
            target_token_indexers: Dict[str, TokenIndexer] = None,
            source_add_start_token: bool = True,
            source_add_end_token: bool = True,
            delimiter: str = "\t",
            EOS: str = "[SEP]",
            source_max_tokens: Optional[int] = None,
            target_max_tokens: Optional[int] = None,
            lazy: bool = False,
            training: bool = True,
            perfect_entity_linking: bool = True,
            delexicalization: bool = False,
            dataset: str = 'graphq',  # ['graphq', 'webq', 'cwq']
            init_var_representation: str = 'utterance',
            eval: bool = False
    ) -> None:
        super().__init__(lazy)
        # self._source_tokenizer = source_tokenizer or (lambda x: x.split())
        self._source_tokenizer = source_tokenizer or SpacyWordSplitter()
        self._target_tokenizer = target_tokenizer or (lambda x: x.replace('(', ' ( ').replace(')', ' ) ').split())
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers
        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._delimiter = delimiter
        self.EOS = EOS
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self._training = training
        self._perfect_el = perfect_entity_linking
        self._delexicalization = delexicalization
        self._dataset = dataset
        self._using_plm = using_plm
        self._init_var_rep = init_var_representation
        self._eval = eval
        if not self._perfect_el:
            if self._dataset == 'grail':
                # with open(path + "/../el_results/grail_test.json") as f:
                with open(path + "/../el_results/grail_dev.json") as f:
                    self._el_results = json.load(f)
            elif self._dataset == 'graphq':
                with open(path + "/../el_results/graphq_test.json") as f:
                    self._el_results = json.load(f)
            elif self._dataset == 'webq':
                with open(path + "/../el_results/webq_test.json") as f:
                    self._el_results = json.load(f)

        if not self._training and self._dataset == "grail":
            self._answer_types = defaultdict(lambda: [])
            with open(path + "/../answer_typing/answer_types_0308.test.txt", 'r') as f:
                for line in f:
                    line = line.replace("\n", '')
                    fields = line.split('\t')
                    for item in fields[1:]:
                        self._answer_types[fields[0]].append(item)

        self._typos = 0
        self._max_target_len = 0

    @overrides
    def _read(self, file_path: str):
        if self._dataset == 'grail':
            constants_path = path + '/../vocabulary/grail/target_tokens.txt'
        elif self._dataset == 'graphq':
            constants_path = path + '/../vocabulary/graphq/target_tokens.txt'
        elif self._dataset == 'webq':
            constants_path = path + '/../vocabulary/webq_full/target_tokens.txt'

        self._schema_constants = set()  # all vocab items loaded from the vocabulary
        for line in open(constants_path):
            self._schema_constants.add(line.replace('\n', ''))

        with open(cached_path(file_path), 'r') as data_file:
            file_contents = json.load(data_file)
            for item in file_contents:
                if item['qid'] in [2102902009000]:  # will exceed maximum length constraint
                    continue

                el_hypo = 0  # hypothesis id
                el_results = self._el_results[str(item['qid'])]
                if len(el_results) > 1:
                    if self._dataset == "webq":
                        time_constraints = {}
                        for er in el_results:
                            if len(el_results[er]) == 1:
                                if isinstance(el_results[er], int):
                                    time_constraints[er] = [str(el_results[er][0])]
                    for er in el_results:
                        if self._dataset != "webq":
                            instance = self.text_to_instance(item, el_results={er: el_results[er]},
                                                            el_hypo=el_hypo)
                        else:
                            if er in time_constraints:
                                continue
                            else:
                                webq_el = {k: v for k, v in time_constraints.items()}
                                webq_el[er] = el_results[er]
                                instance = self.text_to_instance(item, el_results=webq_el,
                                                                 el_hypo=el_hypo)
                        if instance is not None:
                            el_hypo += 1
                            yield instance
                instance = self.text_to_instance(item, el_results=el_results, el_hypo=el_hypo)
                if instance is not None:
                    yield instance

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    @overrides
    def text_to_instance(self,
                         item: Dict = None,
                         logical_forms: List = None,
                         el_results=None,
                         el_hypo=None) -> Instance:  # type: ignore
        if el_hypo is not None:
            qid = MetadataField(str(item['qid']) + '_' + str(el_hypo))
        else:
            qid = MetadataField(item['qid'])

        #  To be consistent with BERT NER, so I can easily get the start and end position of a mention
        item['question'] = ' '.join(word_tokenize(item['question']))
        item['question'] = item['question'].replace('``', '"').replace("''", '"')

        source_string = item['question'].lower()

        if not self._training and self._dataset == "graphq":
            answer_types = self._answer_types[str(item['qid'])]
        else:
            answer_types = None

        if self._dataset in ["graphq", "grail"]:
            if 's_expression' in item:
                target_string = item['s_expression']
            else:
                target_string = None
        else:
            if 's_expression_processed' in item:
                target_string = item['s_expression_processed']
            else:
                target_string = None

        if not self._eval and target_string is None:
            return None

        if self._eval:
            target_string = None

        if self._using_plm:
            tokenized_source = self._source_tokenizer.tokenize(source_string)
        else:
            tokenized_source = [x for x in self._source_tokenizer.split_words(source_string)]
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        if self._source_add_end_token:
            tokenized_source.append(Token(END_SYMBOL))

        # Initially, the key-value map should only contain entities and literals identified in the utterance.
        # It's not necessary to be a map. Instead, it's a list of (variable, (start, end)), keys are just list indexes
        # In the model, the value should be (variable, vector representation) instead
        initial_map = []
        variable_to_id = {}
        variable_id = 0
        if self._perfect_el:
            for node in item['graph_query']['nodes']:
                if node['node_type'] == 'entity':
                    if self._dataset == "webq" and node["implicit"] == 1:
                        continue
                    try:
                        if node['id'] not in variable_to_id:  # For WebQSP there can be repeating nodes.
                            if self._init_var_rep == 'utterance':
                                start, end = self._get_mention_span(
                                    ' '.join(word_tokenize(node['friendly_name'])).lower(), tokenized_source)
                                # initial_map[variable_id] = (node['id'], (start, end))
                                initial_map.append((node['id'], (start, end)))
                            elif self._init_var_rep == 'surface':
                                initial_map.append((node['id'], ' '.join(word_tokenize(node['friendly_name'])).lower()))
                            variable_to_id[node['id']] = variable_id
                            variable_id += 1
                    except UnboundLocalError:
                        # There can be typos...
                        self._typos += 1
                        return None
            if self._dataset == "webq":  # There is no graph query for some manual sparql
                if item['topic_entity'] not in variable_to_id:
                    # print(item['qid'])
                    if self._init_var_rep == 'utterance':
                        try:
                            start, end = self._get_mention_span(
                                ' '.join(word_tokenize(item['topic_entity_name'])).lower(), tokenized_source)
                            initial_map.append((item['topic_entity'], (start, end)))
                        except Exception:
                            return None
                    elif self._init_var_rep == 'surface':
                        initial_map.append(
                            (item['topic_entity'], ' '.join(word_tokenize(item['topic_entity_name'])).lower()))
                    variable_to_id[item['topic_entity']] = variable_id
                    variable_id += 1
        else:
            for mention in el_results:
                for entity in el_results[mention]:
                    try:
                        if self._init_var_rep == 'utterance':
                            start, end = self._get_mention_span(mention.lower(), tokenized_source)
                            initial_map.append((entity, (start, end)))
                        elif self._init_var_rep == 'surface':
                            initial_map.append((entity, mention.lower()))
                        variable_to_id[entity] = variable_id
                        variable_id += 1
                    except UnboundLocalError:
                        print(mention, tokenized_source)
                        pass
        # print(initial_map)

        if self._delexicalization:  # I guarantee that there is no overlap between different mention spans
            assert self._init_var_rep == 'utterance'
            #  list.sort() is stable. What I want to achieve is to first sort with the mention start, and for the same
            # mention span, sort with the score
            initial_map.sort(key=lambda x: x[1][0])
            new_initial_map = []
            to_minus = 0
            previous_start, previous_end = -1, -1
            for i, map_i in enumerate(initial_map):
                if map_i[1][0] == previous_start:
                    start = new_initial_map[i - 1][1][0]
                    if map_i[1][1] != previous_end:
                        print(initial_map)
                        print(item['qid'])
                    assert map_i[1][1] == previous_end  # as we don't allow any overlap or inclusion in mention spans
                    end = new_initial_map[i - 1][1][1]
                else:
                    start = map_i[1][0] - to_minus
                    end = map_i[1][1] - to_minus

                    if not map_i[0][:2] in ["m.", "g."]:
                        pass
                    else:
                        notable_type = get_notable_type(map_i[0])[0].lower()
                        if notable_type == 'entity':  # do nothing
                            pass
                        else:
                            if self._using_plm:
                                tokenized_type = self._source_tokenizer.tokenize(notable_type)
                                tokenized_type = tokenized_type[1:-1]  # strip [CLS] and [SEP]
                            else:
                                tokenized_type = self._source_tokenizer.split_words(notable_type)

                            to_minus += (end - start - len(tokenized_type) + 1)
                            tmp = tokenized_source[end + 1:]
                            tokenized_source = tokenized_source[: end + 1]
                            tokenized_source[start: end + 1] = tokenized_type
                            tokenized_source.extend(tmp)
                            end = start + len(tokenized_type) - 1

                if len(map_i) == 3:
                    new_initial_map.append((map_i[0], (start, end), map_i[2]))
                else:
                    new_initial_map.append((map_i[0], (start, end), 1))  # set score to be constant 1

                previous_start = map_i[1][0]
                previous_end = map_i[1][1]

            # now re-sort it based on score, so entity with higher score will in a sense be prefered in beam search
            new_initial_map.sort(key=lambda x: x[2], reverse=True)
            initial_map = []
            for map_i in new_initial_map:
                initial_map.append((map_i[0], (map_i[1][0], map_i[1][1])))
            original_text = ''
            for token in tokenized_source[1:-1]:
                if len(token.text) > 2 and token.text[:2] == "##" and token.text[2] != "#":
                    original_text += token.text[2:]
                else:
                    original_text += (' ' + token.text)
            original_text = MetadataField(original_text.strip())
            new_variable_to_id = {}
            for i, m in enumerate(initial_map):
                new_variable_to_id[m[0]] = i
            variable_to_id = new_variable_to_id
        else:
            original_text = MetadataField(source_string)

        source_field = TextField(tokenized_source, self._source_token_indexers)

        answers = []
        if 'answer' in item:
            for a in item['answer']:
                answers.append(a["answer_argument"])
        if 'domains' in item:
            domains = item['domains']
        else:
            domains = []

        instance_dict = {"source_tokens": source_field,
                         "original_text": original_text,
                         "initial_map": MetadataField(initial_map),
                         "ids": qid,
                         "answers": MetadataField(answers),
                         "domains": MetadataField(domains),
                         "answer_types": MetadataField(answer_types)}
        if target_string is not None:
            sub_formulas = self._linearize_lisp(target_string, variable_id - 1, variable_to_id)
            target_tokens = []
            constant_or_variable = []  # -1 for constant, id in key-memory map for variable
            for sub_formula in sub_formulas:
                target_tokens.append('(')
                constant_or_variable.append(-1)
                for e in sub_formula:
                    if isinstance(e, list):
                        try:
                            if e[0] != 'R':
                                raise Exception("Unexpected: only R can be nested in a sub formula")
                            else:
                                target_tokens.extend(['(', e[0], e[1], ')'])
                                constant_or_variable.extend([-1, -1, -1, -1])

                        except Exception as e:
                            return None
                    else:
                        try:
                            if e in self._schema_constants:
                                target_tokens.append(e)
                                constant_or_variable.append(-1)
                            else:
                                target_tokens.append('@@UNKNOWN@@')  # doesn't matter what this token is
                                if e[0] != '#':
                                    raise ValueError
                                constant_or_variable.append(int(e[1]))
                        except ValueError:
                            return None

                            target_tokens = target_tokens[:-1]  # abandon the last token
                            # Typos in entity mention can lead to missing items in variable_to_id and init_map
                            pass

                target_tokens.append(')')
                constant_or_variable.append(-1)

            assert len(constant_or_variable) == len(target_tokens)
            tokenized_target = [Token(x) for x in target_tokens]
            if len(tokenized_target) > self._max_target_len:
                self._max_target_len = len(tokenized_target)

            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]
                print("Target exceeds max length!")
            tokenized_target.insert(0, Token(START_SYMBOL))
            constant_or_variable.insert(0, -1)
            tokenized_target.append(Token(END_SYMBOL))
            constant_or_variable.append(-1)

            target_field = TextField(tokenized_target, self._target_token_indexers)
            instance_dict['target_tokens'] = target_field
            instance_dict['constant_or_variable'] = ArrayField(np.asarray(constant_or_variable), padding_value=-1)

            instance_dict["s_expression"] = MetadataField(target_string)

            if self._dataset == 'grail' and 'level' in item:
                instance_dict["level"] = MetadataField(item['level'])

        return Instance(instance_dict)

    def _preprocess_relation_path_for_superlatives(self, expression):
        relations = []
        for element in expression:
            if element == 'JOIN':
                continue
            if isinstance(element, list) and element[0] != 'R':
                assert element[0] == 'JOIN'
                relations.extend(self._preprocess_relation_path_for_superlatives(element))
                continue
            relations.append(element)

        return relations

    def _process_inv_function(self, expression: List):
        # to replace (R XXX) to XXX_inv
        for i, item in enumerate(expression):
            if isinstance(item, list):
                if item[0] == 'R':
                    expression[i] = item[1] + '_inv'
                else:
                    self._process_inv_function(item)

    def _linearize_lisp(self, formula: str, start_id, variable_to_id):
        expression = util.lisp_to_nested_expression(formula)
        self._process_inv_function(expression)  # in this way we can predict R relation in a single step
        # preprocess superlatives to remove JOIN in the second arguments. Note that, the following code only works when
        # superlatives are never nested in other functions
        # During postprocessing, we need to recover it
        if expression[0] in ['ARGMIN', 'ARGMAX']:
            if isinstance(expression[2], list) and expression[2][0] == 'JOIN':
                arg_path = self._preprocess_relation_path_for_superlatives(expression[2])

                expression = expression[:2]
                expression.extend(arg_path)
        sub_formulas = self._linearize_lisp_expression(expression, [start_id])
        for sub_formula in sub_formulas:
            for i, e in enumerate(sub_formula):
                if isinstance(e, str) and e in variable_to_id:
                    sub_formula[i] = '#' + str(variable_to_id[e])
        # print(sub_formulas)
        return sub_formulas

    def _linearize_lisp_expression(self, expression: list, sub_formula_id):
        sub_formulas = []
        for i, e in enumerate(expression):
            # if isinstance(e, list) and e[0] != 'R':
            if isinstance(e, list):
                sub_formulas.extend(self._linearize_lisp_expression(e, sub_formula_id))
                expression[i] = '#' + str(sub_formula_id[0])

        if expression[0] in ['JOIN', 'AND', 'lt', 'le', 'gt', 'ge']:
            tmp = expression[1]
            expression[1] = expression[2]
            expression[2] = tmp

        sub_formulas.append(expression)
        sub_formula_id[0] += 1

        return sub_formulas

    def _same_token(self, token0, token1):
        text0 = token0.text
        text1 = token1.text
        if self.EOS == '</s>':
            if text0[0] == 'Ġ':
                text0 = text0[1:]
            if text1[0] == 'Ġ':
                text1 = text1[1:]

        if text0 in ['"', '``', "''"] and text1 in ['"', '``', "''"]:
            return True

        return text0 == text1

    def _get_mention_span(self, mention, tokenized_source):
        if self._using_plm:
            mention_tokens = self._source_tokenizer.tokenize(mention)
            mention_tokens = mention_tokens[1:-1]
        else:
            mention_tokens = self._source_tokenizer.split_words(mention)
        # print(mention_tokens, tokenized_source)
        flag = False
        for i, token in enumerate(tokenized_source):
            if self._same_token(token, mention_tokens[0]) and i + len(mention_tokens) <= len(tokenized_source):
                flag = True
                for j in range(len(mention_tokens)):
                    if not self._same_token(mention_tokens[j], tokenized_source[i + j]):
                        flag = False
                        break
                if flag:
                    start = i
                    end = i + len(mention_tokens) - 1
                    break

        if not flag and self.EOS == '</s>':  # for roberta, the tokenization is senstive to spaces
            mention_tokens = self._source_tokenizer.tokenize('a ' + mention)
            mention_tokens = mention_tokens[2:-1]

            flag = False
            for i, token in enumerate(tokenized_source):
                if self._same_token(token, mention_tokens[0]) and i + len(mention_tokens) <= len(tokenized_source):
                    flag = True
                    for j in range(len(mention_tokens)):
                        if not self._same_token(mention_tokens[j], tokenized_source[i + j]):
                            flag = False
                            break
                    if flag:
                        start = i
                        end = i + len(mention_tokens) - 1
                        break

        return start, end

