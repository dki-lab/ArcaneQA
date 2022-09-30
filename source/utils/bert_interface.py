from overrides import overrides
from pathlib import Path
from pytorch_transformers.modeling_auto import AutoModel
from pytorch_transformers.modeling_utils import PretrainedConfig
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import get_text_field_mask

path = str(Path(__file__).parent.absolute())


@TokenEmbedder.register("my_pretrained_transformer")
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
    """

    def __init__(self, model_name: str, layers_to_freeze=None, pooling=False) -> None:
        super().__init__()
        self.model_name = model_name
        if "roberta" in model_name:
            self.EOS = 2
        elif "bert" in model_name:
            self.EOS = 102
        # config = PretrainedConfig.from_json_file(path + "/../bert_configs/debug.json")
        # self.transformer_model = AutoModel.from_pretrained(model_name, config=config)
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

        if layers_to_freeze is not None:
            modules = [self.transformer_model.embeddings,
                       *self.transformer_model.encoder.layer[:layers_to_freeze]]  # Replace 5 by what you want
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        self._pooling = pooling

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:  # type: ignore
        if "roberta" in self.model_name:
            # In RoBerta's vocabulary, padding is not mapped to 0 as allennlp's default setting,
            # instead, it is mapped to 1, so here we temporarily replace 1 to 0, while 0 stands for <s>,
            # we just replace 0 to a random non-padding id, e.g., 23 here
            attention_mask = get_text_field_mask({'bert': token_ids.masked_fill(token_ids == 0, 23)
                                                 .masked_fill(token_ids == 1, 0)})
            # roberta doesn't have type ids, it distinguish the first and second sentence only based on </s>
            token_type_ids = None
        elif "bert" in self.model_name:
            attention_mask = get_text_field_mask({'bert': token_ids})
            token_type_ids = self.get_type_ids(token_ids)
        # attention_mask = None
        # pylint: disable=arguments-differ
        # position_ids = self.get_position_ids(token_ids).to(token_ids.device)
        # token_type_ids = None
        position_ids = None

        outputs = self.transformer_model(token_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                                         attention_mask=attention_mask)

        if self._pooling:
            return outputs[1]
        else:
            return outputs[0]

    def get_type_ids(self, token_ids: torch.LongTensor):
        type_ids = torch.zeros_like(token_ids)
        num_seq, max_len = token_ids.shape
        for i in range(num_seq):
            for j in range(max_len):
                if token_ids[i][j] == self.EOS:  # id of [SEP] or </s>'s first occurence
                    break
            type_ids[i][j + 1:] = 1
        return type_ids

    def get_position_ids(self, token_ids: torch.LongTensor):
        position_ids = []
        num_seq, max_len = token_ids.shape
        for i in range(num_seq):
            position_ids_i = []
            next_id = 0
            # first_sep = True
            for j in range(max_len):
                position_ids_i.append(next_id)
                # if token_ids[i][j] == 102 and first_sep:
                if token_ids[i][j] == 102:
                    next_id = 0
                    # first_sep = False  # in case [SEP] is used as delimiter for schema constants
                else:
                    next_id += 1

            position_ids.append(position_ids_i)
        return torch.LongTensor(position_ids)