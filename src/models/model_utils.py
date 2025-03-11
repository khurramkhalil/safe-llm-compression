import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.utils.prune as prune
from fnmatch import fnmatch
import copy
import os

# Global model instance to avoid repeated deep copies
comp_model = None

# def load_base_model(model_name="deepseek-ai/deepseek-llm-7b-base", save_dir="./saved_models/"):
#     """Load the base model and tokenizer, save locally, and return them."""
#     os.makedirs(save_dir, exist_ok=True)
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_name, torch_dtype=torch.float16, device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     base_model.save_pretrained(os.path.join(save_dir, "deepseek-7b-local"))
#     tokenizer.save_pretrained(os.path.join(save_dir, "deepseek-7b-local"))
#     original_size = sum(p.numel() * 2 for p in base_model.parameters()) / (1024 ** 2)  # FP16: 2 bytes per param
#     return base_model, tokenizer, original_size

def load_base_model(model_name, save_dir="./saved_models/"):
    """Load the base model and tokenizer."""
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    original_size = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 * 1024)  # Size in MB
    return model, tokenizer, original_size

def quantize_tensor(tensor, bits):
    """Quantize a tensor to the specified number of bits."""
    max_val = torch.max(torch.abs(tensor))
    scale = (2 ** (bits - 1) - 1) / max_val if max_val > 0 else 1.0
    q_param = torch.clamp(torch.round(tensor * scale), -(2**(bits-1)), 2**(bits-1) - 1).to(torch.int32)
    return q_param, scale

def log_quantized_size(model, layer_bits):
    """Calculate the size of the quantized model in MB."""
    total_bytes = 0
    for name, data in model.quantized_state.items():
        bits = next((b for p, b in layer_bits.items() if fnmatch(name, p)), 12)
        total_bytes += data["q_param"].numel() * bits // 8
    return total_bytes / (1024 ** 2)

def get_layer_groups(model):
    """Identify linear layers in the model for configuration."""
    return [{"pattern": name + ".*", "name": name.replace(".", "_")}
            for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

def reset_model(base_model):
    """Reset the global compressed model to the base model's state."""
    global comp_model
    if comp_model is None:
        comp_model = copy.deepcopy(base_model)  # Initial copy only once
    else:
        with torch.no_grad():
            for param, base_param in zip(comp_model.parameters(), base_model.parameters()):
                param.data.copy_(base_param.data)
        for module in comp_model.modules():
            if isinstance(module, torch.nn.Linear) and prune.is_pruned(module):
                prune.remove(module, 'weight')
    comp_model.quantized_state = {}
    return comp_model

def apply_config(model, config):
    """Apply pruning and quantization to the model based on the configuration."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_config = next((c for c in config["layers"] if fnmatch(name + ".weight", c["pattern"])), {"prune": 0.0})
            if layer_config["prune"] > 0:
                prune.l1_unstructured(module, name="weight", amount=float(layer_config["prune"]))
    
    quantized_state = {}
    for name, param in model.named_parameters():
        layer_config = next((c for c in config["layers"] if fnmatch(name, c["pattern"])), {"bits": 12})
        q_param, scale = quantize_tensor(param.data, layer_config["bits"])
        quantized_state[name] = {"q_param": q_param, "scale": scale}
        param.data = q_param.to(torch.float16) / scale  # Keep FP16
    model.quantized_state = quantized_state
    return model

# def forward_with_signals_batched(model, input_ids):
#     """Run a batched forward pass and return probabilities, hidden states, and attentions."""
#     with torch.no_grad():
#         output = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True)
#         probs = torch.softmax(output.logits, dim=-1)
#         hidden_states = output.hidden_states[-1]
#         attention_matrices = {f"layer_{i}": attn for i, attn in enumerate(output.attentions)}
#         return {"probs": probs, "attention_matrices": attention_matrices, "hidden_states": hidden_states}

def forward_with_signals_batched(model, input_ids):
    """Forward pass with signals."""
    outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
    return {
        "probs": outputs.logits.softmax(dim=-1),
        "attention_matrices": outputs.attentions,
        "hidden_states": outputs.hidden_states
    }


def log_quantized_size(model, layer_bits):
    """Estimate quantized model size."""
    total_size = 0
    for name, param in model.named_parameters():
        for pattern, bits in layer_bits.items():
            if pattern in name:
                total_size += param.nelement() * (bits / 8) / (1024 * 1024)  # Bytes to MB
                break
        else:
            total_size += param.element_size() * param.nelement() / (1024 * 1024)
    return total_size