import warnings
from typing import Optional, Tuple

import torch
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertAttention,
    BertEmbeddings,
    MaskedLMOutput,
    PreTrainedModel,
)
from transformers.pytorch_utils import apply_chunking_to_forward

from .lsh.hash import STR2HASH
from .lsh.node import LSHLinear
from .lsh.sampling import STR2SAMPLING
import torch.nn as nn

from .configuration_lsh import LSHConfig
import dataclasses


def init_weights(config, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


@dataclasses.dataclass
class PretrainingOutput:
    hidden_states: Optional[torch.Tensor]
    mlm_loss: Optional[torch.Tensor]
    seq_loss: Optional[torch.Tensor]
    total_loss: Optional[torch.Tensor]


def construct_lsh_module(config: LSHConfig, input_size: int, output_size: int):
    hash_module = STR2HASH[config.lsh_hash_function]
    sampling_module = STR2SAMPLING[config.sampling_function]
    sampling_parameters = {k: v for k, v in config.to_dict().items() if k.startswith("sampling_")}
    return LSHLinear(
        input_size,
        output_size,
        config.lsh_tables,
        config.lsh_functions,
        hash_module,
        sampling_module,
        sampling_parameters,
    )


def construct_linear_module(config: LSHConfig, input_size: int, output_size: int, lsh=False):
    if lsh:
        return construct_lsh_module(config, input_size, output_size)
    return torch.nn.Linear(input_size, output_size)


class LMHead(torch.nn.Module):
    def __init__(self, config: LSHConfig):
        super().__init__()
        self.dense = construct_linear_module(
            config, config.hidden_size, config.hidden_size, "cls" in config.lsh_location
        )
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.transform = construct_linear_module(
            config, config.hidden_size, config.hidden_size, "cls" in config.lsh_location
        )
        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SequenceClassificationHead(torch.nn.Module):
    def __init__(self, config: LSHConfig):
        super().__init__()
        self.dense = construct_linear_module(
            config, config.hidden_size, config.hidden_size, "cls" in config.lsh_location
        )
        self.activation = torch.nn.Tanh()
        self.seq_relationship = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class Intermediate(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = construct_linear_module(
            config, config.hidden_size, config.intermediate_size, "mlp" in config.lsh_location
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = construct_linear_module(
            config, config.intermediate_size, config.hidden_size, "mlp" in config.lsh_location
        )
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(torch.nn.Module):
    def __init__(self, config: LSHConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder(torch.nn.Module):
    def __init__(self, config: LSHConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.size()[:-1]
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, dtype=hidden_states.dtype
        )

        hidden_states = (hidden_states,)
        for _, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states[0], extended_attention_mask)
        return hidden_states

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int], dtype: torch.float = None
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


class LSHPretrainedModel(PreTrainedModel):
    config_class = LSHConfig
    base_model_prefix = "lsh"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Encoder):
            module.gradient_checkpointing = value


class LSHModel(LSHPretrainedModel):
    def __init__(self, config: LSHConfig):
        super().__init__(config)
        self.config = config
        self.embedding = BertEmbeddings(config)
        self.encoder = Encoder(config)
        init_weights(config, self)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, token_type_ids: torch.Tensor = None
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        if len(input_ids.shape) == 1:
            input_ids = torch.unsqueeze(input_ids, 0)

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embedding, "token_type_ids"):
                buffered_token_type_ids = self.embedding.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        hidden_states = self.embedding(input_ids, token_type_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)

        return BaseModelOutputWithPoolingAndCrossAttentions(hidden_states, None, None, None, None, None)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings


class LSHForMaskedLM(LSHPretrainedModel):
    def __init__(self, config: LSHConfig):
        super().__init__(config)
        self.config = config
        self.model = LSHModel(config)
        self.lm_head = LMHead(config)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        init_weights(config, self)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> MaskedLMOutput:
        output = self.model(input_ids, attention_mask, token_type_ids)
        prediction_scores = self.lm_head(output[0][0])

        if labels is not None:
            masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            masked_lm_loss = None

        return MaskedLMOutput(masked_lm_loss, prediction_scores, output.hidden_states, output.attentions)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def _tie_embeddings(self):
        input_embeddings = self.get_input_embeddings()
        output_embeddings = self.get_output_embeddings()

        output_embeddings.weight = input_embeddings.weight
        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
