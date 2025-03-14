import torch
import numpy as np
import pandas as pd
from src.models.model_utils import apply_config, forward_with_signals_batched
from ..stl.stl_monitoring import monitor_stl_signals
from ..utils.common_utils import clear_gpu_memory

iteration = 0
best_objective = float('inf')
best_params = None
falsified_configs = set()

def objective_function(params, base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir="./saved_models/", model_name="unknown"):
    global iteration, best_objective, best_params, falsified_configs
    iteration += 1
    print(f"Iteration {iteration}: Starting with params (first 6): {params[:6]}")
    
    config = {"layers": []}
    for i, group in enumerate(layer_groups):
        bits = int(max(4, min(16, params[i * 2])))
        prune_amount = float(max(0.0, min(0.5, params[i * 2 + 1])))
        config["layers"].append({"pattern": group["pattern"], "bits": bits, "prune": prune_amount})
    
    try:
        comp_model = apply_config(base_model, config)
        comp_model.cuda()
        
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(
            [tokenizer(seq["input"], return_tensors="pt", truncation=True, max_length=50).input_ids[0] for seq in dataset_subset],
            batch_first=True, padding_value=tokenizer.pad_token_id
        ).cuda()
        
        comp_signals = forward_with_signals_batched(comp_model, input_ids_batch)
        
        max_new_tokens = max([len(gt) for gt in ground_truth_cache if gt is not None]) if any(ground_truth_cache) else 20
        with torch.no_grad():
            gen_output = comp_model.generate(
                input_ids_batch,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,  # Enable sampling for diversity
                top_k=50,        # Limit to top 50 tokens
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,  # Reduced from 2.0
                min_length=input_ids_batch.shape[1] + 1,
                return_dict_in_generate=True,
                output_scores=False,
            )
            generated_ids_batch = gen_output.sequences
        
        total_robustness = 0.0
        falsification_count = 0
        sample_count = len(dataset_subset)
        robustness_dict = {}
        
        for i in range(sample_count):
            base_signals = base_signals_cache[i]
            comp_signals_batch = {
                "probs": comp_signals["probs"][i:i+1],
                "attention_matrices": {k: v[i:i+1] for k, v in comp_signals["attention_matrices"].items()},
                "hidden_states": comp_signals["hidden_states"][i:i+1]
            }
            ground_truth_ids = ground_truth_cache[i]
            input_len = dataset_subset[i]["input_len"]
            generated_ids = generated_ids_batch[i, input_len:].tolist() if ground_truth_ids else None
            
            print(f"Debug: Sample {i}, Ground Truth IDs={ground_truth_ids[:5] if ground_truth_ids else None}..., Generated IDs={generated_ids[:5] if generated_ids else None}...")
            if ground_truth_ids and generated_ids:
                gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                gt_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(f"Sample {i}: Input={dataset_subset[i]['input']}, Ground Truth='{gt_text}', Generated='{gen_text}'")
            else:
                print(f"Sample {i}: Skipping print - No generated IDs or ground truth")
            
            robustness, falsified = monitor_stl_signals(base_signals, comp_signals_batch, specs, ground_truth_ids, input_len, generated_ids)
            print(f"Debug: Sample {i}, Raw Robustness={robustness}")  # Log raw values
            for k in robustness:
                if not isinstance(robustness[k], (int, float)) or np.isnan(robustness[k]) or np.isinf(robustness[k]):
                    robustness[k] = -10.0 if k == 'seq_coh' else 0.0  # Reduced penalty
            robustness_dict[i] = robustness
            total_robustness += sum(robustness.values())
            falsification_count += any(falsified.values())
            print(f"Iteration {iteration}: Sample {i}, Robustness={robustness}, Falsified={falsified}")
        
        compressed_size = 0
        for i, p in enumerate(comp_model.parameters()):
            if i < len(config["layers"]):
                compressed_size += p.numel() * (config["layers"][i]["bits"] / 8)
            else:
                compressed_size += p.numel() * (config["layers"][-1]["bits"] / 8)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        objective = total_robustness / sample_count - compression_ratio
        
        print(f"Iteration {iteration}: Objective={objective:.4f}, Falsified Samples={falsification_count}")
        clear_gpu_memory()
        return objective, robustness_dict, falsification_count
    
    except Exception as e:
        print(f"Iteration {iteration}: Error - {str(e)}")
        return 1e6, {}, 0

def run_optimization(base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir="./saved_models/", model_name="unknown"):
    global best_objective, best_params
    from bayes_opt import BayesianOptimization
    
    pbounds = {f'p{i}': (4, 16) if i % 2 == 0 else (0, 0.5) for i in range(len(layer_groups) * 2)}
    
    def wrapped_objective(**params):
        obj, _, _ = objective_function(
            [params[f'p{i}'] for i in range(len(layer_groups) * 2)], 
            base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, 
            ground_truth_cache, specs, original_size, save_dir, model_name
        )
        return obj
    
    optimizer = BayesianOptimization(
        f=wrapped_objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    logs = pd.DataFrame(columns=['iteration', 'params', 'objective', 'falsified_samples', 'robustness'])
    
    # Run initial points
    optimizer.maximize(init_points=5, n_iter=0)
    for i in range(5):
        params = list(optimizer.res[i]['params'].values())
        objective, robustness, falsified_count = objective_function(
            params, base_model, dataset_subset, tokenizer, layer_groups, 
            base_signals_cache, ground_truth_cache, specs, original_size, save_dir, model_name
        )
        logs.loc[len(logs)] = [iteration, params, objective, falsified_count, robustness]
        print(f"Initial Point {i+1}: Params (first 6): {params[:6]}..., Falsified Samples: {falsified_count}, Robustness: {list(robustness.values())[-1]}")
        if objective < best_objective:
            best_objective = objective
            best_params = params
    
    # Run remaining iterations
    optimizer.maximize(init_points=0, n_iter=15)
    for i in range(5, 20):  # 5 initial + 15 iterations
        params = list(optimizer.res[i]['params'].values())
        objective, robustness, falsified_count = objective_function(
            params, base_model, dataset_subset, tokenizer, layer_groups, 
            base_signals_cache, ground_truth_cache, specs, original_size, save_dir, model_name
        )
        logs.loc[len(logs)] = [iteration, params, objective, falsified_count, robustness]
        print(f"Iteration {i-4}: Params (first 6): {params[:6]}..., Falsified Samples: {falsified_count}, Robustness: {list(robustness.values())[-1]}")
        if objective < best_objective:
            best_objective = objective
            best_params = params
    
    logs.to_csv(f"{save_dir}/{model_name}_optimization_log.csv", index=False)
    print(f"Optimization complete. Best objective: {best_objective:.4f}, Best params (first 6): {best_params[:6] if best_params else None}...")
    return best_params, logs

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure pad token is set
    dataset_subset = [
        {"input": "where is zimbabwe located in the world map", "input_len": 9},
        {"input": "where did the beatles final live performance take place", "input_len": 11},
        {"input": "where is arkansas river located on a map", "input_len": 9}
    ]
    ground_truth_cache = [
        tokenizer.encode("in southern Africa, between the Zambezi and Limpopo Rivers", return_tensors="pt")[0].tolist(),
        tokenizer.encode("the roof of the headquarters of the band's multimedia corporation", return_tensors="pt")[0].tolist(),
        tokenizer.encode("The Arkansas River flows through Colorado, Kansas, Oklahoma", return_tensors="pt")[0].tolist()
    ]
    layer_groups = [{"pattern": "layer.*"}]
    base_signals_cache = [{"probs": torch.randn(1, 50, 256000), "hidden_states": torch.randn(1, 50, 2304), "attention_matrices": {}}] * 3
    specs = {}
    original_size = 1e6
    best_params, logs = run_optimization(model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size)