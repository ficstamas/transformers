from transformers import AutoModelForMaskedLM, AutoConfig


# cfg = AutoConfig.from_pretrained("configs/config_lsh_mlp-only.json")
model = AutoModelForMaskedLM.from_pretrained("test_lsh_model/")

# model.save_pretrained("test_lsh_model/")
breakpoint()