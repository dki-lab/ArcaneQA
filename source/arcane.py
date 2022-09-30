import time
import numpy as np
from typing import Dict, List, Tuple, Union

from utils.logical_form_util import same_logical_form, lisp_to_sparql, binary_nesting
from utils.sparql_executer import execute_query
from utils.my_pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from utils.kb_engine import KBEngine

import numpy
import math
import random
import re
import json
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import Average
from utils.arcane_beam_search import Arcane_BeamSearch
from allennlp.semparse.util import lisp_to_nested_expression

import functools

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

MAX_VARIABLES_NUM = 200


@Model.register("arcane_seq2seq")
class Arcane_Seq2Seq(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    # Parameters

    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_namespace : ``str``, optional (default = "tokens")
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function : ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015](https://arxiv.org/abs/1506.03099).
    """

    def __init__(
            self,
            vocab: Vocabulary,
            source_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            max_decoding_steps: int,
            attention: Attention = None,
            target_word_embedder: TextFieldEmbedder = None,  # only for GloVe
            attention_function: SimilarityFunction = None,
            beam_size: int = None,
            target_namespace: str = "tokens",
            target_embedding_dim: int = None,
            scheduled_sampling_ratio: float = 0.0,
            dropout: float = 0.0,
            add_noise: bool = False,
            # two options: 'utterance' and 'surface'
            init_var_representation: str = 'utterance',
            dataset: str = 'graphq',
            using_plm: bool = False,
            plm_dim: int = 768,
            num_constants_per_group: int = 40,  # only when using_plm is true
            delimiter: str = ';',  # only when using_plm is true
            EOS: str = '[SEP]',
            eval: bool = False
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self._num_constants = len(self.vocab._index_to_token[target_namespace])

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        # self._entity_names = json.load(open('cache/entity_names.json'))
        self._entity_names = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 10
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = Arcane_BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size, length_normalization=False,
            vocab=self.vocab
        )

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        if using_plm:
            # This can only be hard coded here. AllenNLP config file doesn't support having tokenizer and
            # indexer under 'model' parameter
            self.EOS = EOS
            self._plm_dim = plm_dim
            if self.EOS != '</s>':
                self._source_tokenizer = PretrainedTransformerTokenizer(
                    model_name="bert-base-uncased",
                    do_lowercase=True
                )
            else:
                self._source_tokenizer = PretrainedTransformerTokenizer(
                    model_name="roberta-base",
                    do_lowercase=True  # this argument is useless for roberta
                )

            self._num_constants_per_group = num_constants_per_group
            self._delimiter = delimiter

        if not using_plm:
            self._target_word_embedder = target_word_embedder

        self._dropout = torch.nn.Dropout(p=dropout)

        self._add_noise = add_noise

        self._init_var_rep = init_var_representation

        self._dataset = dataset

        self._using_plm = using_plm

        self._eval = eval

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        self._exact_match = Average()

        self._exact_match_k = Average()

        self._em_zero = Average()

        self._F1 = Average()

        self._MRR_k = Average()

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError(
                    "You can only specify an attention module or an "
                    "attention function, but not both."
                )
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        self._encoder_input_projection_layer = nn.Linear(source_embedder.get_output_dim(),
                                                         self._encoder.get_input_dim())
        projected_target_embedding_dim = self._encoder.get_input_dim()
        self._target_embedding_projection_layer = nn.Linear(target_embedding_dim, projected_target_embedding_dim)

        if self._init_var_rep == "utterance":
            assert self._encoder_output_dim == projected_target_embedding_dim

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = projected_target_embedding_dim + self._encoder_output_dim
            self._output_projection_layer = Linear(self._decoder_output_dim + self._encoder_output_dim,
                                                   projected_target_embedding_dim)
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = projected_target_embedding_dim
            self._output_projection_layer = Linear(self._decoder_output_dim, projected_target_embedding_dim)

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._output_embedding = None
        self._values = None
        self._device = None
        self._batch_size = None

        self._epoch_num = 0

        self._uncovered = [0]

        self._kb_engine = KBEngine(dataset=dataset, MAX_VARIABLES_NUM=MAX_VARIABLES_NUM)
        self._relations, self._classes, self._attributes = self._kb_engine.get_vocab()

        self._times = []

    # For previous output, there is only one argument last_predictions to describe that, which means we should always
    # concatenate the constants and variables into one joint vector
    def take_step(
            self,
            last_predictions: torch.Tensor,
            state: Dict[str, Union[torch.Tensor, List]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        # Parameters

        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        # Returns

        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state, representative_mask = self._prepare_output_projections(last_predictions, state)

        vocab_mask = state["vocab_mask"]

        # output_projections.masked_fill_(vocab_mask == 0, -1e32)
        output_projections.masked_fill_(representative_mask == 0, -1e32)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        class_log_probabilities.masked_fill_(vocab_mask == 0, -1e32)

        for i in range(len(class_log_probabilities)):
            if (vocab_mask[i] == 0).all().item():  # no admissible action
                class_log_probabilities[i] = -1e32

        # # TODO: this is temporary for debugging overfitting variable selection
        # class_log_probabilities[:, self._num_constants:][vocab_mask[:, self._num_constants:] == 1] = 0

        return class_log_probabilities, state

    # @timer
    @overrides
    def forward(
            self,  # type: ignore
            source_tokens: Dict[str, torch.LongTensor],
            original_text: List[str] = None,  # only used for PLM
            target_tokens: Dict[str, torch.LongTensor] = None,
            constant_or_variable: torch.LongTensor = None,
            initial_map=None,
            ids=None,
            s_expression=None,  # only for easier debugging visualization
            answers=None,
            domains=None,
            answer_types=None,
            level=None,
            epoch_num=None  # use epoch_num[0] to get the integer epoch number
    ) -> Dict[str, torch.Tensor]:

        """
        Make foward pass with decoder logic for producing the entire target sequence.

        # Parameters

        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        # Returns

        Dict[str, torch.Tensor]
        """

        self._encoder._module.flatten_parameters()

        self._kb_engine.set_training(self.training)

        # cache the sparql executions to file every epoch (this is clumsy, but I don't want to modify the framework
        # code of allennlp)
        if epoch_num is not None and self._epoch_num != epoch_num[0]:
            self._uncovered = [0]
            self._epoch_num = epoch_num[0]
            self._kb_engine._cache.cache_results()

        device = source_tokens["tokens"].device
        self._device = device
        self._batch_size = source_tokens["tokens"].shape[0]
        if not self._using_plm:
            output_vocab, output_vocab_mask \
                = self._make_embedder_input(
                self._get_target_tokens(torch.arange(0, self.vocab.get_vocab_size(self._target_namespace))))

            # (num_classes, decoder_output_dim)
            output_embedding = self._compute_target_embedding(output_vocab, output_vocab_mask)
            # (decoder_output_dim, num_classes)
            self._output_embedding = output_embedding.transpose(0, 1)

        if target_tokens and not self._eval:
            state = self._encode(source_tokens)
            state["original_text"] = original_text
            state["answer_types"] = answer_types
            state["domains"] = domains
            # (batch_size, 1)
            state["batch_id"] = torch.arange(0, self._batch_size).unsqueeze(-1).to(self._device)
            assert target_tokens["tokens"].shape[1] == constant_or_variable.shape[1]
            target_tokens["tokens"] = target_tokens["tokens"].float()
            target_tokens["tokens"][constant_or_variable != -1] = \
                self._num_constants + constant_or_variable[constant_or_variable != -1]
            state = self._init_decoder_state(state, initial_map)
            # for v: [[], []]  the first list stores the relation path, while the second list stores the
            # induced variables
            state["derivations"] = [{v: {v: [[]]} for v in range(len(variables))} for variables in state["variables"]]

            output_dict = self._forward_loop(state, target_tokens)

            if self.training:
                for i, prediction in enumerate(output_dict['predictions']):
                    self._exact_match(self._compute_exact_match_train(prediction, target_tokens["tokens"][i]))

        else:
            output_dict = {}

        if not self.training:
            start_time = time.time()
            # Reinitialize for inference
            state = self._encode(source_tokens)
            state["original_text"] = original_text
            state["answer_types"] = answer_types
            state["domains"] = domains
            # (batch_size, 1)
            state["batch_id"] = torch.arange(0, self._batch_size).unsqueeze(-1).to(self._device)
            state = self._init_decoder_state(state, initial_map)

            state["derivations"] = [{v: {v: [[]]} for v in range(len(variables))} for variables in state["variables"]]

            #  AllenNLP's beam search returns no more than beam_size of finished states
            predictions = self._forward_beam_search(state)

            output_dict.update(predictions)
            # self._output_predictions(predictions['predictions'])

            if not self._eval:
                for i, prediction in enumerate(predictions['predictions']):
                    if i in output_dict["uncovered_ids"]:
                        unco = True
                    else:
                        unco = False
                    em = self._compute_exact_match(prediction[0],  # 0 means the top one from beam
                                                   target_tokens["tokens"][i],
                                                   source_tokens["tokens"][i],
                                                   initial_map[i],
                                                   ids[i],
                                                   predictions["class_log_probabilities"][i][0],
                                                   unco=unco)
                    self._exact_match(em)
                    if self._dataset == "graphq" and level[i] == 'zero-shot':
                        self._em_zero(em)
                    if em == 1:
                        self._F1(1)
                    else:
                        # self._F1(self._compute_F1(prediction[0], initial_map[i], answers[i]))
                        self._F1(0)  # TODO: this is temporary for fast iteration
                    # print("\nutterance:", self._get_utterance(source_tokens["tokens"][i]))
                    # print("target:", self._get_logical_form(target_tokens["tokens"][i][1:], initial_map[i]))
                    # for j in range(10):
                    #     print(
                    #         f"predicted {j}: {self._get_logical_form(prediction[j], initial_map[i])} {predictions['class_log_probabilities'][i][j]}")

                for i, prediction_k in enumerate(predictions['predictions']):
                    em_k, mrr_k = self._compute_exact_match_k(prediction_k,
                                                              target_tokens["tokens"][i],
                                                              initial_map[i])
                    self._exact_match_k(em_k)
                    self._MRR_k(mrr_k)

            output_dict["initial_map"] = initial_map
            output_dict['ids'] = ids

            self._times.append(time.time() - start_time)
            times = np.array(self._times)
            print(len(self._times), np.mean(times), np.std(times))

        return output_dict

    @DeprecationWarning
    def _output_predictions(self, predictions):
        """
        Out put the best predicted logical form for each batch instance
        :param predictions: (batch_size, beam_size, num_decoding_steps)
        :return:
        """
        for prediction in predictions:
            logical_form = []
            for token_id in prediction[0]:
                logical_form.append(self.vocab.get_token_from_index(token_id.item(), self._target_namespace))
            rtn = logical_form[0]
            for i in range(1, len(logical_form)):
                if logical_form[i] == '@end@':
                    break
                if logical_form[i - 1] == '(' or logical_form[i] == ')':
                    rtn += logical_form[i]
                else:
                    rtn += ' '
                    rtn += logical_form[i]
            print(rtn)

    def _get_utterance(self, token_ids) -> str:
        question = []
        for token_id in token_ids:
            if self._using_plm:
                token = self.vocab.get_token_from_index(token_id.item(), "bert")
            else:
                token = self.vocab.get_token_from_index(token_id.item(), "source_tokens")
            if token == '@end@':
                break
            question.append(token)

        return ' '.join(question[:])

    def _ids_to_tokens(self, token_ids):  # TODO: _ids_to_tokens(self, inv_indicator)
        tokens = []
        for token_id in token_ids:
            if token_id.item() < self._num_constants:
                tokens.append(self.vocab.get_token_from_index(token_id.item(), self._target_namespace))
            else:
                tokens.append('#' + str(int(token_id.item()) - self._num_constants))

        return tokens

    def _get_partial_logical_form(self, token_ids) -> str:
        logical_form = self._ids_to_tokens(token_ids)
        rtn = logical_form[0]
        for i in range(1, len(logical_form)):
            if logical_form[i] == '@end@':
                break
            if logical_form[i - 1] == '(' or logical_form[i] == ')':
                rtn += logical_form[i]
            else:
                rtn += ' '
                rtn += logical_form[i]

        return rtn

    # Note that, during training, the constraints are based on the ground truth instead of the model's own prediction,
    # so the predicted logical form during training can be pretty weird
    def _get_logical_form(self, token_ids, initial_map=None) -> str:
        variable_map = []
        for item in initial_map:
            variable_map.append(item[0])

        logical_form = []
        sub_formulas = []
        for token_id in token_ids:
            if token_id.item() < self._num_constants:
                logical_form.append(self.vocab.get_token_from_index(token_id.item(), self._target_namespace))
            else:
                logical_form.append('#' + str(int(token_id.item()) - self._num_constants))
        if logical_form[0] == START_SYMBOL:
            sub_formula = logical_form[1]
        else:
            sub_formula = logical_form[0]
        # R_flag = False

        for i in range(1, len(logical_form)):
            if logical_form[i] == '@end@':
                break
            if logical_form[i - 1] == '(':
                sub_formula += logical_form[i]
            elif logical_form[i] == ')':
                sub_formula += logical_form[i]
                sub_formulas.append(sub_formula)
                sub_formula = ''
            else:
                if len(sub_formula) > 0:
                    sub_formula += ' '
                if logical_form[i][-4:] != '_inv':
                    sub_formula += logical_form[i]
                else:
                    sub_formula += '(R ' + logical_form[i][:-4] + ')'

        for sub_formula in sub_formulas:
            try:
                expression = lisp_to_nested_expression(sub_formula)
            except Exception:
                # not sure why
                # This happens when do compute_exact_match_k, because there may not be enough faithful queries to
                # fill the beam
                continue
            if expression[0] in ['JOIN', 'AND', 'lt', 'le', 'gt', 'ge']:
                tmp = expression[1]
                try:
                    expression[1] = expression[2]
                    expression[2] = tmp
                except IndexError:
                    print("unexpected:", sub_formula)
                    continue
            for i, element in enumerate(expression):
                if isinstance(element, list):
                    expression[i] = f"({element[0]} {element[1]})"

            if expression[0] in ['ARGMAX', 'ARGMIN'] and len(expression) > 3:  # post-processing for superlatives
                sub_formula = "(" + expression[0] + " " + expression[1] + " " + binary_nesting("JOIN",
                                                                                               expression[2:]) + ")"
            elif expression[0] == "CONS":  # post-processing for constraints in webq
                try:
                    sub_formula = f"(AND {expression[1]} (JOIN {expression[2]} {expression[3]}))"
                except IndexError:
                    print("unexpected:", sub_formula)
                    continue
            else:
                sub_formula = "(" + " ".join(expression) + ")"

            variables = re.findall(r"#[0-9]*", sub_formula)
            for variable in variables:
                vid = int(variable[1:])
                try:
                    sub_formula = sub_formula.replace(variable, variable_map[vid])
                except Exception:
                    pass
            variable_map.append(sub_formula)

        return sub_formula  # the last sub formula is the complete formula

    def _compute_F1(self, predicted, initial_map, answers):
        try:
            sparql_query = lisp_to_sparql(self._get_logical_form(predicted, initial_map))
            denotation = set(execute_query(sparql_query))
            correct = denotation.intersection(set(answers))
            precision = len(correct) / len(denotation)
            recall = len(correct) / len(answers)

            return (2 * precision * recall) / (precision + recall)

        except:
            return 0

    def _compute_exact_match_train(self,
                                   predicted: torch.Tensor,
                                   target: torch.Tensor):
        predicted = self._get_partial_logical_form(predicted)
        target = self._get_partial_logical_form(target[1:])

        # if predicted != target:
        #     print("\npredicted:", predicted)
        #     print("target:", target)

        if predicted == target:
            return 1
        else:
            return 0

    def _compute_exact_match(self,
                             predicted: torch.Tensor,
                             target: torch.Tensor,
                             source: torch.Tensor,
                             initial_map,
                             qid,
                             probability: torch.Tensor = None,
                             unco: bool = False) -> int:
        predicted = self._get_logical_form(predicted, initial_map)
        target = self._get_logical_form(target[1:], initial_map)  # omit the start symbol

        if same_logical_form(predicted, target):
            return 1
        else:
            return 0

    def _compute_exact_match_k(self,
                               predicted_k: torch.Tensor,
                               target: torch.Tensor,
                               initial_map) -> int:
        target_logical_form = self._get_logical_form(target[1:], initial_map)  # omit the start symbol
        for i, predicted in enumerate(predicted_k):
            predicted_logical_form = self._get_logical_form(predicted, initial_map)
            if same_logical_form(predicted_logical_form, target_logical_form):
                return 1, 1. / (i + 1)

        return 0, 0

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        initial_map = output_dict["initial_map"]
        ids = output_dict['ids']
        scores = output_dict["class_log_probabilities"]

        all_predicted_lfs = []
        all_predicted_answers = []
        all_scores = []
        for indices, ini_map, qid, score in zip(predicted_indices, initial_map, ids, scores):  # for each batch instance
            # indices = indices[0]
            predicted_lf = ''
            max_score = -1e32
            for beam_id, indices_i in enumerate(indices):  # for each beam instance
                try:
                    predicted_lf = self._get_logical_form(indices_i, ini_map)
                except Exception:
                    pass
                denotation = []

                if not (predicted_lf.__contains__("ARG") and len(ini_map) == 0 and len(predicted_lf) > 250):
                    try:
                        sparql_query = lisp_to_sparql(predicted_lf)
                        denotation.extend(execute_query(sparql_query))
                    except Exception:
                        pass
                if len(denotation) > 0:
                    max_score = score[beam_id]
                    break

            if len(ini_map) > 1:
                flag = False
                optimal_lf = indices_i
                for lf in indices[1:]:
                    if self._better_entity(optimal_lf, lf):
                        optimal_lf = lf
                        flag = True
                if flag:
                    predicted_lf = self._get_logical_form(optimal_lf, ini_map)

                    if predicted_lf.__contains__("ARG") and len(ini_map) == 0 and len(predicted_lf) > 250:
                        denotation = []
                    else:
                        denotation = []
                        try:
                            sparql_query = lisp_to_sparql(predicted_lf)
                            denotation.extend(execute_query(sparql_query))
                        except Exception:
                            pass

            all_predicted_answers.append(denotation)
            all_predicted_lfs.append(predicted_lf)
            all_scores.append(max_score)

        rtn = {}
        rtn['qid'] = ids
        rtn['logical_form'] = all_predicted_lfs
        rtn['answer'] = all_predicted_answers
        rtn['score'] = all_scores

        # if (int(ids[0].split('-')[1]) + 1) % 200 == 0:
        #     self._kb_engine._cache.cache_results()

        return rtn

    def _better_entity(self, lf0, lf1):  # whether lf1 has the same structure as lf0, but with better entity
        if len(lf0) != len(lf1):
            return False
        else:
            flag = False
            for i in range(len(lf0)):
                if lf0[i] != lf1[i]:
                    if lf0[i] < self._num_constants or lf1[i] < self._num_constants:
                        return False
                    else:
                        if lf0[i] > lf1[i]:  # Initial map is in decreasing order
                            flag = True
            return flag

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Besides encoding the utterance, we also want to initialize our key-value map here.
        The original key-value map from dataset reader only provides us the span information of initial variables,
        however, in th model, we want to have a different structure: @# -> (variable, tensor),
        where variable is a set of entities / literals, while tensor is the representation of the variable based on
        which we do decoding.
        :param source_tokens:
        :param initial_map:
        :return:
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)

        embedded_input = self._encoder_input_projection_layer(embedded_input)

        # shape: (batch_size, max_input_sequence_length)
        if self._using_plm and self.EOS == '</s>':
            # In RoBerta's vocabulary, padding is not mapped to 0 as allennlp's default setting,
            # instead, it is mapped to 1, so here we temporarily replace 1 to 0, while 0 stands for <s>,
            # we just replace 0 to a random non-padding id, e.g., 23 here
            source_mask = util.get_text_field_mask(
                {'tokens': source_tokens['tokens'].masked_fill(source_tokens['tokens'] == 0, 23)
                    .masked_fill(source_tokens['tokens'] == 1, 0)})   # not masked_fill_
        else:
            source_mask = util.get_text_field_mask(source_tokens)  # mask to be handled by encoder
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        outputs = self._encoder(embedded_input, source_mask)
        encoder_outputs = self._dropout(outputs)

        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _init_decoder_state(self, state: Dict[str, torch.Tensor],
                            # [[(mid, (start, end))]]
                            initial_map: List[List[Tuple[str, Tuple[int, int]]]]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            state["encoder_outputs"], state["source_mask"], self._encoder.is_bidirectional()
        )
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self._decoder_output_dim
        )

        # (batch_size, num_of_variables, dim),  num_of_variables is never gonna exceed 10 (actually smaller than 5)
        variable_embedding = torch.zeros(self._batch_size, MAX_VARIABLES_NUM, state["encoder_outputs"].shape[-1]).to(
            self._device)

        variables = []
        for i in range(len(initial_map)):
            map_i = initial_map[i]
            variables_i = []
            for j in range(len(map_i)):
                if self._init_var_rep == 'utterance':
                    variable_embedding[i][j] = state["encoder_outputs"][i][map_i[j][1][0]:map_i[j][1][1] + 1].mean(0)
                elif self._init_var_rep == 'surface':
                    words = map_i[j][1].split()
                    for word in words:
                        word_id = torch.tensor([self.vocab.get_token_index(word, "source_tokens")],
                                               device=self._device)
                        word_embedding = self._source_embedder({"tokens": word_id}).squeeze(0)
                        variable_embedding[i][j] += word_embedding
                    variable_embedding[i][j] /= len(map_i[j][1].split())

                if map_i[j][0][:2] in {'g.', 'm.'}:  # entity:
                    variables_i.append({map_i[j][0]})  # a set of entities
                else:  # literal
                    variables_i.append(map_i[j][0])

            variables.append(variables_i)

        state["variables"] = variables
        state["variable_embedding"] = variable_embedding

        state["variable_num"] = torch.Tensor([[len(variables_i)] for variables_i in variables]).to(self._device)

        # state.pop("two_tokens", [])
        state.pop("predictions", [])

        state["arg_mode"] = [False for _ in range(batch_size)]
        state["arg_variables"] = [{} for _ in range(batch_size)]
        state["arg_class"] = [None for _ in range(batch_size)]

        state["initial_map"] = initial_map

        return state

    # @timer
    def _forward_loop(
            self,
            state: Dict[str, torch.Tensor],
            target_tokens: Dict[str, torch.LongTensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_target_sequence_length)
        targets = target_tokens["tokens"].long()

        _, target_sequence_length = targets.size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.  (In fact, I beleive this comment from ai2 is incorrect,
        # the real reason is we don't have to process the <sos> token)
        num_decoding_steps = target_sequence_length - 1

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []

        for timestep in range(num_decoding_steps):
            # shape: (batch_size,)
            input_choices = targets[:, timestep]

            # This has already been handled before passing to forward_loop. No need to repeat here.
            # input_choices[constant_flag != -1] = self._num_constants + constant_flag[constant_flag != -1]

            if self.training:
                next_tokens = targets[:, timestep + 1]
                # shape: (batch_size, num_classes)
                output_projections, state, _ = self._prepare_output_projections(input_choices, state,
                                                                                next_tokens=next_tokens)
            else:
                # shape: (batch_size, num_classes)
                output_projections, state, _ = self._prepare_output_projections(input_choices, state)

            # print("targets:", targets[:, timestep + 1])
            # print("vocab items:", (state["vocab_mask"] == 1).nonzero())
            # This is used to check cause for infinite loss
            # for i in range(len(targets)):
            #     #  targets[i, timestep + 1].item() != 0 means not padding. We don't consider padding here
            #     if state["vocab_mask"][i][targets[i, timestep + 1]] != 1 and targets[i, timestep + 1].item() != 0:
            #         print("targets: ", targets[i][:timestep + 2])
            # This is used to check some unexpected behavior (i.e., predictions not conform with constrained decoding):
            # for i in range(len(targets)):
            #     if state["vocab_mask"][i][self.vocab.get_token_index(')', self._target_namespace)] == 1 and \
            #         input_choices[i] == self.vocab.get_token_index('(', self._target_namespace):
            #         print('wtf')

            # apply the vocab mask
            output_projections.masked_fill_(state["vocab_mask"] == 0, -1e32)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            uncovered_ids = []
            loss = self._get_loss(logits, targets, target_mask, 'batch', self._uncovered, uncovered_ids)

            output_dict["loss"] = loss  # Here is the only place that loss being calculated
            output_dict["uncovered_ids"] = uncovered_ids

        return output_dict

    def _forward_beam_search(self,
                             state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # can be used to compute exact match
        # shape (log_probabilities): (batch_size, beam_size), the probability of generating
        # the associated sequence
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor],
                                    next_tokens: torch.Tensor = None  # This is only used for training
                                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        This is also where you update the state for next step!!!

        Inputs are the same as for `take_step()`.
        """

        group_size = len(last_predictions)

        assert group_size == state["variable_embedding"].shape[0]

        # update arg_variables based on relation, when arg_mode is true and the input of this step if relation
        for i in range(len(last_predictions)):
            if state["arg_mode"][i]:
                last_predicted_i = last_predictions[i].int().item()
                if last_predicted_i >= self._num_constants:
                    # Unexpected. Might happen at the early stage of beam search
                    pass
                else:
                    token_i = self.vocab.get_token_from_index(last_predicted_i, self._target_namespace)
                    if token_i in self._attributes:
                        # if the relation is an attribute then do nothing
                        pass
                    elif token_i in self._relations:
                        if state["arg_class"][i] is not None:
                            state["arg_class"][i] = self._kb_engine._relation_r[token_i]
                        else:
                            state["arg_variables"][i] = self._kb_engine.execute_JOIN(state["arg_variables"][i],
                                                                                    token_i + '_inv')
                    elif token_i[-4:] == '_inv' and token_i[:-4] in self._relations:
                        if state["arg_class"][i] is not None:
                            state["arg_class"][i] = self._kb_engine._relation_d[token_i[:-4]]
                        else:
                            state["arg_variables"][i] = self._kb_engine.execute_JOIN(state["arg_variables"][i],
                                                                                    token_i)

        # initialize arg_variables when the input of this step is a variable (or class) and the previous step is ARG
        for i in range(len(last_predictions)):
            if "predictions" in state and state["predictions"][i][-1].item() < self._num_constants:
                if self.vocab.get_token_from_index(state["predictions"][i][-1].int().item(),
                                                   self._target_namespace).__contains__("ARG"):
                    state["arg_mode"][i] = True
                    if last_predictions[i].int().item() >= self._num_constants:
                        state["arg_variables"][i] = state["variables"][i][
                            int(last_predictions[i].int().item() - self._num_constants)]
                    else:
                        state["arg_class"][i] = self.vocab.get_token_from_index(last_predictions[i].int().item(),
                                                                                self._target_namespace)

        for i in range(group_size):
            if "predictions" in state and state["predictions"][i][-1].item() < self._num_constants:
                # if last_predictions[i].item() < self._num_constants:
                if self.vocab.get_token_from_index(state["predictions"][i][-1].int().item(),
                                                   self._target_namespace) == ')' and self.vocab.get_token_from_index(
                    last_predictions[i].int().item(), self._target_namespace) != END_SYMBOL:
                    new_var_id = len(state["variables"][i])
                    # The following statement would lead to in-place operation that cannot be tracked during bp.
                    # state["variable_embedding"][i][new_var_id] = state["decoder_hidden"][i]
                    # Instead, do the following:
                    variable_embedding = state["variable_embedding"].clone()
                    # TODO: replace variable representation here
                    variable_embedding[i][new_var_id] = state["decoder_hidden"][i]
                    state["variable_embedding"] = variable_embedding
                    state["variables"][i].append(self._execute_partial_program(state["predictions"][i],
                                                                               state["variables"][i],
                                                                               state["derivations"][i]))
                    state["variable_num"][i] = state["variable_num"][i] + 1

        # We also need to store the predictions during the process for partial program execution.
        # We can also perform comparison between the predictions in state and the final predictions provided
        # by beam search for sanity check (for inference, since for training we always feed golden supervision).
        # Update predictions base on the prediction from last step
        if "predictions" not in state:
            predictions = []
            for i in range(len(last_predictions)):
                assert last_predictions[i] == self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
                predictions.append(torch.Tensor([[last_predictions[i]]]).to(self._device))
            state["predictions"] = torch.cat(predictions, dim=0).long()
        else:
            state["predictions"] = torch.cat([state["predictions"], last_predictions.unsqueeze(-1)], dim=1)

        # (group_size, num_constants)
        vocab_mask = torch.zeros(group_size, self._num_constants).to(self._device)
        # (group_size, num_variables)    # 10 is the upperbound of number of variables
        variable_mask = torch.zeros(group_size, MAX_VARIABLES_NUM).to(self._device)

        # (group_size, num_constants)
        representative_mask = torch.zeros(group_size, self._num_constants).to(self._device)

        if self._using_plm:
            if "output_embeddings" not in state:
                # Initialize as whatever values, doesn't matter
                state["output_embeddings"] = torch.zeros(group_size, self._plm_dim, self._num_constants, device=self._device)

        for i in range(group_size):
            admissible_constants, admissible_variables, representative_items = \
                self._kb_engine.get_admissible_actions(self._ids_to_tokens(state["predictions"][i]),
                                                      state["variables"][i],
                                                      len(state["initial_map"][i]),
                                                      state["arg_mode"][i],
                                                      state["arg_variables"][i],
                                                      state["arg_class"][i],
                                                      add_noise=self._add_noise,
                                                      domains=state["domains"][i],
                                                      answer_types=state["answer_types"][i],
                                                      derivations=state["derivations"][i],
                                                      initial_map = state["initial_map"][i])

            # we don't want to miss any training examples
            if self.training and next_tokens is not None:
                next_token_i = next_tokens[i]
                if next_token_i.int().item() >= self._num_constants:
                    next_var = next_token_i.int().item() - self._num_constants
                    if next_var not in admissible_variables:
                        admissible_variables.append(next_var)
                else:
                    next_constant = self.vocab.get_token_from_index(next_token_i.int().item(), self._target_namespace)
                    if next_constant not in admissible_constants:
                        admissible_constants.append(next_constant)

            all_constants = []
            all_constants.extend(admissible_constants)
            all_constants.extend(representative_items)
            # TODO: to think about how to represent utterance when no constant is admissible
            if self._using_plm:
                if len(admissible_constants) > 0:  # maybe set to > 1
                    utterance = state["original_text"][i]
                    # dynamic_encoder_outputs, constants_outputs \
                    #     = self._dynamic_encoding(utterance, admissible_constants)
                    dynamic_encoder_outputs, constants_outputs = self._dynamic_encoding(utterance, all_constants)
                    state["encoder_outputs"] = state["encoder_outputs"].clone()
                    state["encoder_outputs"][i][:dynamic_encoder_outputs.shape[0]] = dynamic_encoder_outputs
                    # update state["encoder_outputs"] and state["output_embeddings"], state["source_mask"] still the same
                    state["output_embeddings"] = state["output_embeddings"].clone()
                    # for j, constant in enumerate(admissible_constants):
                    for j, constant in enumerate(all_constants):
                        constant_index = self.vocab._token_to_index[self._target_namespace][constant]
                        # (group_size, self._plm_dim, num_constants)
                        state["output_embeddings"][i][:, constant_index] = \
                            constants_outputs[j]

            for constant in admissible_constants:
                try:
                    vocab_mask[i][self.vocab._token_to_index[self._target_namespace][constant]] = 1
                except KeyError:
                    # It's possible that some relations from base, user are not in our vocabulary
                    # print('key error:', constant)
                    pass

            for constant in all_constants:
                try:
                    representative_mask[i][self.vocab._token_to_index[self._target_namespace][constant]] = 1
                except KeyError:
                    # It's possible that some relations from base, user are not in our vocabulary
                    # print('key error:', constant)
                    pass

            for v in admissible_variables:
                variable_mask[i][v] = 1

        # (group_size, num_constants + num_variables)
        state["vocab_mask"] = torch.cat((vocab_mask, variable_mask), dim=1)

        representative_mask = torch.cat((representative_mask, variable_mask), dim=1)

        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # (group_size, embedding_dim, num_values)
        variable_embedding = state["variable_embedding"]

        if self._using_plm:
            embedded_input = state["output_embeddings"][torch.arange(0, group_size, device=self._device), :,
                             last_predictions.masked_fill(last_predictions >= self._num_constants,
                                                          0)]  # whatever fill value, will be replaced anyway
        else:
            # TODO: No need to compute this again, can directly take from output_embeddings
            converted_input, mask = self._make_embedder_input(self._get_target_tokens(last_predictions))
            embedded_input = self._compute_target_embedding(converted_input, mask)

        embedded_input = self.replace_variable_embeddings(embedded_input, last_predictions, state)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(
                decoder_hidden, encoder_outputs, source_mask
            )

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # TODO: use attention also for logits projection not just input feeding
        if self._attention:
            decoder_output = torch.cat((attended_input, decoder_hidden), -1)
        else:
            decoder_output = decoder_hidden

        if self._using_plm:
            output_embedding = state["output_embeddings"]
        else:
            output_embedding = self._output_embedding

        projected_hidden = self._output_projection_layer(
            decoder_output)  # one more linear layer before computing logits

        if self._using_plm:
            # (group_size, 1, hidden_size)
            projected_hidden = projected_hidden.unsqueeze(1)

            # (group_size, 1, num_schema_items)
            output_projections = torch.bmm(projected_hidden, output_embedding)
            output_projections = output_projections.squeeze(1)
        else:
            # (batch_size, num_schema_items)
            output_projections = torch.mm(projected_hidden, output_embedding)

            projected_hidden = projected_hidden.unsqueeze(1)

        output_projections_values = torch.bmm(projected_hidden, variable_embedding.transpose(1, 2))
        # (batch_size, num_values)
        output_projections_values = output_projections_values.squeeze(1)

        output_projections = torch.cat((output_projections, output_projections_values), dim=-1)

        return output_projections, state, representative_mask

    def _compute_target_embedding(self,
                                  x: torch.Tensor,
                                  mask: torch.Tensor,
                                  pooling: str = None) -> torch.Tensor:
        pooling = pooling or 'mean'
        x = x.to(self._device)
        mask = mask.to(self._device)

        # (batch_size, num_of_words, embedding_dim)
        embeddings = self._target_word_embedder({"tokens": x})  # Note here "tokens" is specified to match the embedder
        # (batch_size, num_of_words, embedding_dim)
        mask = (mask.unsqueeze(-1)).expand(-1, -1, embeddings.shape[-1])
        # (batch_size, num_of_words, embedding_dim)
        embeddings = embeddings * mask

        if pooling == 'sum':
            # (batch_size, embedding_dim)
            embeddings = embeddings.sum(1)
        elif pooling == 'mean':
            mask = mask[:, :, 0].sum(1).unsqueeze(1)
            embeddings = embeddings.sum(1)
            embeddings = embeddings / mask

        embeddings = self._target_embedding_projection_layer(embeddings)

        # (batch_size, embedding_dim)
        return embeddings

    # Only process one instance a time. No batched operation for now
    # TODO: change to batch operation and see whether can be faster
    def _dynamic_encoding(self, utterance: str,
                          constants: List[str]):
        constants = constants.copy()
        # If not training, add random constants here. Those random noise will be masked. Just to make sure the
        # distribution is the same for training and inference
        # if not self.training and self._add_noise:
        #     if constants[0] in self._relations:
        #         constants.extend(random.sample(self._relations, 30))
        #     elif constants[0] in self._classes:
        #         constants.extend(random.sample(self._classes, 30))

        # If not adding noise, num_chunks is typically one.
        # num_chunks = len(constants) // self._num_constants_per_group + 1
        num_chunks = math.ceil(len(constants) / self._num_constants_per_group)
        # print("num chunks:", num_chunks)
        concat_strings = ['' for _ in range(num_chunks)]

        for i in range(num_chunks):
            if (i + 1) * self._num_constants_per_group <= len(constants):
                right_index = (i + 1) * self._num_constants_per_group
            else:
                right_index = len(constants)
            for constant in constants[i * self._num_constants_per_group: right_index]:
                if self._dataset in ["webq", "cwq"] and constant in self._kb_engine._cons_ids:
                    # concat_strings[i] += ' '.join(
                    #     re.split('\.|_', self._kb_engine._cons_ids[constant])) + self._delimiter
                    # for roberta
                    concat_strings[i] += ' '.join(
                        re.split('\.|_', self._kb_engine._cons_ids[constant])) + ' ' + self._delimiter + ' '
                else:
                    # concat_strings[i] += ' '.join(re.split('\.|_', constant)) + self._delimiter
                    # for roberta
                    concat_strings[i] += ' '.join(re.split('\.|_', constant)) + ' ' + self._delimiter + ' '

        len_utterance = len(self._source_tokenizer.tokenize(utterance))

        tokenized_sources = [self._source_tokenizer.tokenize(utterance + self.EOS + concat_string)
                             for concat_string in concat_strings]

        end = []
        start = []
        for tokenized_source in tokenized_sources:
            flag = False
            for i, token in enumerate(tokenized_source):
                if self.EOS == '</s>':
                    if flag and str(token) == 'Ġ' + self._delimiter:
                        end.append(i - 1)
                        start.append(i + 1)
                elif self.EOS == '[SEP]':
                    if flag and str(token) == self._delimiter:
                        end.append(i - 1)
                        start.append(i + 1)
                if str(token) == self.EOS:
                    if not flag:
                        start.append(i + 1)
                    flag = True

            start = start[:-1]  # ignore the last delimiter

        bert_input = self._make_bert_input(tokenized_sources)
        # (num_chunks, max_len, self._plm_dim)
        bert_output = self._source_embedder({'tokens': bert_input})
        embedded_utterance = torch.mean(bert_output[:, :len_utterance], dim=0)
        # (1, len_utterance, self._plm_dim)
        embedded_utterance = embedded_utterance.unsqueeze(0)

        # (1, len_utterance, hidden_size)
        dynamic_encoder_outputs = self._dropout(
            self._encoder(embedded_utterance, embedded_utterance.new_ones(1, len_utterance)))
        # self._encoder(embedded_utterance, embedded_utterance.new_ones(1, len_utterance))
        dynamic_encoder_outputs = dynamic_encoder_outputs.squeeze(0)

        if len(constants) != len(start):
            print("haha")

        assert len(end) == len(start)
        constant_outputs = embedded_utterance.new_zeros(len(constants), self._plm_dim)
        for i in range(len(constants)):
            constant_outputs[i] = \
                torch.mean(bert_output[i // self._num_constants_per_group][start[i]:end[i] + 1], dim=0)

        return dynamic_encoder_outputs, constant_outputs

    # Only process one instance a time. No batched operation for now
    def _make_bert_input(self, tokenized_sources: List[List[str]]):
        max_len = 0
        for tokenized_source in tokenized_sources:
            if len(tokenized_source) > max_len:
                max_len = len(tokenized_source)

        if self.EOS == '[SEP]':
            # [PAD] is 0
            bert_input = torch.zeros(len(tokenized_sources), max_len, device=self._device)
        else:  # roberta
            # [PAD] is 1
            bert_input = torch.ones(len(tokenized_sources), max_len, device=self._device)
        for i, tokenized_source in enumerate(tokenized_sources):
            for j, token in enumerate(tokenized_source):
                #  Here namespace "bert" is consistent with the config for indexer
                bert_input[i][j] = self.vocab.get_token_index(token.text, namespace="bert")

        return bert_input.long()

    def _prepare_attended_input(
            self,
            decoder_hidden_state: torch.LongTensor = None,
            encoder_outputs: torch.LongTensor = None,
            encoder_outputs_mask: torch.LongTensor = None,
    ) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(
            logits: torch.FloatTensor,
            targets: torch.LongTensor,
            target_mask: torch.LongTensor,
            average: str = "batch",
            uncovered=None,
            uncovered_ids=None
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # # shape: (batch_size, num_decoding_steps)
        # relevant_targets = targets[:, 1:].contiguous()
        #
        # # shape: (batch_size, num_decoding_steps)
        # relevant_mask = target_mask[:, 1:].contiguous()
        #
        # return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

        # Instead of using the interface provided by util.sequence_cross_entropy_with_logits, we define the computation
        # here to make it more flexible.

        # shape: (batch_size, num_decoding_steps)
        targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        weights = target_mask[:, 1:].contiguous()

        # make sure weights are float
        weights = weights.float()
        # sum all dim except batch
        non_batch_dims = tuple(range(1, len(weights.shape)))
        # shape : (batch_size,)
        weights_batch_sum = weights.sum(dim=non_batch_dims)
        # shape : (batch * sequence_length, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()

        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood * weights

        if average == "batch":
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
            num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)

            # print("loss: ", per_batch_loss)
            num_non_empty_sequences = num_non_empty_sequences - (per_batch_loss > 1e15).sum().item()
            uncovered[0] = uncovered[0] + (per_batch_loss > 1e15).sum().item()
            for i in range(per_batch_loss.shape[0]):
                if per_batch_loss[i] > 1e15:
                    uncovered_ids.append(i)
            per_batch_loss.masked_fill_(per_batch_loss > 1e15, 0)

            # this would lead to nan loss for num_non_empty_sequences = 0
            # return per_batch_loss.sum() / (num_non_empty_sequences)
            return per_batch_loss.sum() / (num_non_empty_sequences + 1e-13)
        elif average == "token":
            return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
        else:
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
            return per_batch_loss

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # reset is set to be True by default in trainer
        all_metrics: Dict[str, float] = {}
        all_metrics['example_count'] = self._exact_match._count
        all_metrics['exact_match_count'] = self._exact_match._total_value
        all_metrics['exact_match'] = self._exact_match.get_metric(reset)
        all_metrics['exact_match_k'] = self._exact_match_k.get_metric(reset)
        if self._dataset == "graphq":
            all_metrics['em_zero'] = self._em_zero.get_metric(reset)
        all_metrics['F1'] = self._F1.get_metric(reset)
        all_metrics['MRR_k'] = self._MRR_k.get_metric(reset)
        return all_metrics

    # varibles has length batch_size, each item in it corresponds to a list of variables for each instance
    def _get_variable_mask(self, variables: List):
        max_len = 0
        for variables_i in variables:
            if len(variables_i) > max_len:
                max_len = len(variables_i)
        variable_mask = torch.zeros(self._batch_size, max_len).to(self._device)

        for i, variables_i in enumerate(variables):
            variable_mask[i][: len(variables_i)] = 1

        return variable_mask

    def _get_target_tokens(self, x):
        tokens = []
        for i, id in enumerate(x):
            if id.item() < self._num_constants:
                token = self.vocab._index_to_token[self._target_namespace][id.item()]
            else:
                token = 'variable'  # whatever, will be ignored (replaced) anyway

            tokens.append(token)
        return tokens

    def replace_variable_embeddings(self, embedded_input, last_predictions, state):
        assert len(last_predictions) == state["variable_embedding"].shape[0]
        for i in range(len(last_predictions)):
            if last_predictions[i] >= self._num_constants:
                embedded_input[i] = state["variable_embedding"][i][
                    last_predictions[i].int() - self._num_constants]

        return embedded_input

    def _make_embedder_input(self, x):
        """
        Convert a list of logical constant indexes into an input to the word-level embedder
        :param x: (group_size, )
        :return: converted_input: (group_size, num_words), mask: (group_size, num_words)
        Here mask use 1 to indicate being used and 0 for masked, which is different from
        vocab_mask
        """
        group_size = len(x)
        tokens_list = []
        max_len = 0
        for token in x:
            # if len(token) > 2 and  token[:2] == 'm.':
            #     print(token)
            token_words = re.split('[._ ]', token)
            # token_words = token.split('.')[-1].split('_')
            tokens_list.append(token_words)
            max_len = max(max_len, len(token_words))

        mask = torch.zeros(group_size, max_len)
        converted_input = []
        for i, token_words in enumerate(tokens_list):
            # if token_words[0] == 'm':
            #     print(token_words)
            word_ids = []
            for word in token_words:
                if word in self.vocab._token_to_index['tgt_words']:
                    word_ids.append(self.vocab._token_to_index['tgt_words'][word])
                else:
                    word_ids.append(self.vocab._token_to_index['tgt_words']['@@UNKNOWN@@'])
            mask[i][:len(word_ids)] = 1
            for _ in range(max_len - len(word_ids)):
                word_ids.append(0)
            converted_input.append(word_ids)

        converted_input = torch.tensor(converted_input)

        return converted_input, mask

    # Since it's unnecessary to execute the final program, and ARGMAX, ARGMIN, COUNT can only be the most outer one,
    # so there is no need to implement the interpreter for these functions.
    def _execute_partial_program(self, predictions, variables, derivations):
        '''
        predictions record the whole history of decoding (in tensor), the last sub formula should be
        retrieved from it (based on the position of last '(') before execution
        :param predictions:
        :return: a new variable (i.e., a set of entities)
        '''
        rtn = set()
        derivations_to_update = []
        try:
            program = self._get_partial_logical_form(predictions)
            if program.__contains__("@@UNKNOWN@@") or program.__contains__("@@PADDING@@"):
                # this happens when there is no admissible actions. And due to the implementation of log_softmax on
                # cuda. In this case, log_softmax assign 0 to all dimensions
                return rtn
            # position of last (, also notice that we don't treat R as an stand-alone executable function
            position = len(program) - program.replace("(R", '[R')[::-1].index('(') - 1
            sub_program = program[position:]
            expression = lisp_to_nested_expression(sub_program)
            # print("sub program: ", expression)
            assert expression[1][0] == '#'  # only consider JOIN and AND for now
            arg1 = variables[int(expression[1][1:])]
            if isinstance(expression[2], str) and expression[2][0] == '#':
                # I incorrectly used variables[int(expression[1][1:])] before...
                arg2 = variables[int(expression[2][1:])]
            else:
                arg2 = expression[2]
            if expression[0] == 'AND':
                rtn = self._kb_engine.execute_AND(arg1, arg2)
                if expression[2][0] == '#':
                    for k in derivations[int(expression[1][1:])]:
                        derivations[len(variables)] = {}
                        derivations[len(variables)][k] = [[]]
                        derivations[len(variables)][k][0].extend(derivations[int(expression[1][1:])][k][0])
                    for k in derivations[int(expression[2][1:])]:
                        derivations[len(variables)] = {}
                        derivations[len(variables)][k] = [[]]
                        derivations[len(variables)][k][0].extend(derivations[int(expression[2][1:])][k][0])
            elif expression[0] == 'JOIN':
                rtn = self._kb_engine.execute_JOIN(arg1, arg2)
                if arg2[-4:] == '_inv':
                    for k in derivations[int(expression[1][1:])]:
                        derivations[len(variables)] = {}
                        derivations[len(variables)][k] = [[]]
                        derivations[len(variables)][k][0].extend(derivations[int(expression[1][1:])][k][0])
                        derivations[len(variables)][k][0].append(':' + arg2[:-4])
                else:
                    for k in derivations[int(expression[1][1:])]:
                        derivations[len(variables)] = {}
                        derivations[len(variables)][k] = [[]]
                        derivations[len(variables)][k][0].extend(derivations[int(expression[1][1:])][k][0])
                        derivations[len(variables)][k][0].append('^:' + arg2)

            elif expression[0] == 'TC':
                rtn = self._kb_engine.execute_TC(arg1, expression[2], expression[3])
            elif expression[0] == 'CONS':
                #  I don't think those implict constraints in webq can help to reduce search space too much
                #  so directly return the original variable here
                rtn = arg1
                for k in derivations[int(expression[1][1:])]:
                    derivations[len(variables)] = {}
                    derivations[len(variables)][k] = [[]]
                    derivations[len(variables)][k][0].extend(derivations[int(expression[1][1:])][k][0])
            elif expression[0] in ['ARGMAX', 'ARGMIN']:
                #  Same reason as CONS
                rtn = arg1
                for k in derivations[int(expression[1][1:])]:
                    derivations[len(variables)] = {}
                    derivations[len(variables)][k] = [[]]
                    derivations[len(variables)][k][0].extend(derivations[int(expression[1][1:])][k][0])
            elif expression[0] in ['le', 'ge', 'lt', 'gt']:
                comparators = {'le': "<=", 'ge': ">=", 'lt': "<", 'gt': ">"}
                rtn = self._kb_engine.execute_Comparative(arg1, arg2, expression[0])
                if arg2[-4:] == '_inv':
                    for k in derivations[int(expression[1][1:])]:
                        derivations[len(variables)] = {}
                        derivations[len(variables)][k] = [[]]
                        derivations[len(variables)][k][0].extend(derivations[int(expression[1][1:])][k][0])
                        derivations[len(variables)][k][0].append(':' + arg2[:-4])
                        derivations[len(variables)][k].append(comparators[expression[0]])
                else:
                    for k in derivations[int(expression[1][1:])]:
                        derivations[len(variables)] = {}
                        derivations[len(variables)][k] = [[]]
                        derivations[len(variables)][k][0].extend(derivations[int(expression[1][1:])][k][0])
                        derivations[len(variables)][k][0].append('^:' + arg2)
                        derivations[len(variables)][k].append(comparators[expression[0]])

            else:  # superlative and count functions are not nested
                pass
        except Exception:
            # This might happen at the early state of beam search, i.e., there are no enough admissible sequences to
            # fill the beam. Specifically, for the first step, there will be some relation constants included in the
            # beam, and based on our rules, relations can only be followed by a ')'.
            # topk in PyTorch is not sort stable. But sometimes it appears to be sort stable and in that case there
            # will be less error partial program. It's pretty fishy.
            # print(("Error executing partial program!", program, variables))
            return set()

        return set(list(rtn))

