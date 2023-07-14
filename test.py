from transformers.models.lsh import LSHForMaskedLM, LSHConfig
from transformers import AutoModelForMaskedLM, AutoConfig


cfg = AutoConfig.from_pretrained("configs/config_lsh_mlp-only.json")
model = AutoModelForMaskedLM.from_config(cfg)

out = model(**model.dummy_inputs)

breakpoint()