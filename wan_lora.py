import os
import torch
from safetensors import safe_open
from loguru import logger
import gc
from functools import lru_cache

@lru_cache(maxsize=None)
def GET_DTYPE():
    RUNNING_FLAG = os.getenv("DTYPE")
    return RUNNING_FLAG

class WanLoraWrapper:
    def __init__(self, wan_model):
        self.model = wan_model
        self.lora_metadata = {}
        self.override_dict = {}  # On CPU

    def load_lora(self, lora_path, lora_name=None):
        if lora_name is None:
            lora_name = os.path.basename(lora_path).split(".")[0]

        if lora_name in self.lora_metadata:
            logger.info(f"LoRA {lora_name} already loaded, skipping...")
            return lora_name

        self.lora_metadata[lora_name] = {"path": lora_path}
        logger.info(f"Registered LoRA metadata for: {lora_name} from {lora_path}")

        return lora_name

    def _load_lora_file(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(torch.bfloat16) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_name, alpha=1.0):
        if lora_name not in self.lora_metadata:
            logger.info(f"LoRA {lora_name} not found. Please load it first.")

        # if hasattr(self.model, "current_lora") and self.model.current_lora:
        #     self.remove_lora()
        # import ipdb
        # ipdb.set_trace()
        # if not hasattr(self.model, "original_weight_dict"):
        self.model.original_weight_dict = {}
        for k, v in self.model.named_parameters():
            self.model.original_weight_dict[k] = v


        lora_weights = self._load_lora_file(self.lora_metadata[lora_name]["path"])
        weight_dict = self.model.original_weight_dict
        self._apply_lora_weights(weight_dict, lora_weights, alpha)
        # self.model._init_weights(weight_dict)

        logger.info(f"Applied LoRA: {lora_name} with alpha={alpha}")
        return True


    @torch.no_grad()
    def _apply_lora_weights(self, weight_dict, lora_weights, alpha):
        lora_pairs = {}
        prefix = "diffusion_model."

        for key in lora_weights.keys():
            if key.endswith("lora_down.weight") and key.startswith(prefix):
                base_name = key[len(prefix) :].replace("lora_down.weight", "weight")
                b_key = key.replace("lora_down.weight", "lora_up.weight")
                if b_key in lora_weights:
                    lora_pairs[base_name] = (key, b_key)
            elif key.endswith("diff_b") and key.startswith(prefix):
                base_name = key[len(prefix) :].replace("diff_b", "bias")
                lora_pairs[base_name] = (key)
            elif key.endswith("diff") and key.startswith(prefix):
                base_name = key[len(prefix) :].replace("diff", "weight")
                lora_pairs[base_name] = (key)

        applied_count = 0
        for name, param in weight_dict.items():
            if name in lora_pairs:
                if name not in self.override_dict:
                    self.override_dict[name] = param.detach().clone().cpu()

                if len(lora_pairs[name])==2:
                    name_lora_A, name_lora_B = lora_pairs[name]
                    lora_A = lora_weights[name_lora_A].to(param.device, param.dtype)
                    lora_B = lora_weights[name_lora_B].to(param.device, param.dtype)
                    delta = torch.matmul(lora_B, lora_A) * alpha
                    param.add_(delta)
                else:
                    name_lora = lora_pairs[name]
                    delta = lora_weights[name_lora]* alpha
                    param.add_(delta.to(param.device, param.dtype))
                applied_count += 1


        logger.info(f"Applied {applied_count} LoRA weight adjustments")
        if applied_count == 0:
            logger.info(
                "Warning: No LoRA weights were applied. Expected naming conventions: 'diffusion_model.<layer_name>.lora_A.weight' and 'diffusion_model.<layer_name>.lora_B.weight'. Please verify the LoRA weight file."
            )


    def list_loaded_loras(self):
        return list(self.lora_metadata.keys())

    def get_current_lora(self):
        return self.model.current_lora