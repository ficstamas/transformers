from transformers import PretrainedConfig


class LSHConfig(PretrainedConfig):
    model_type = "transformer_structured"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        max_position_embeddings=512,
        type_vocab_size=1,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        classifier_dropout=None,
        attention_type="bert",
        lsh_location="mlp",
        lsh_tables=30,
        lsh_functions=7,
        lsh_hash_function="simhash",
        sampling_function="vanilla",
        sampling_num_target_neurons=128,
        tie_embeddings=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        # bert or reformer (lsh attention)
        self.attention_type = attention_type
        # mlp, attention, mlp_and_attention
        self.lsh_location = lsh_location
        self.lsh_tables = lsh_tables
        self.lsh_functions = lsh_functions
        self.lsh_hash_function = lsh_hash_function
        self.sampling_function = sampling_function
        self.sampling_num_target_neurons = sampling_num_target_neurons
        self.tie_embeddings = tie_embeddings
