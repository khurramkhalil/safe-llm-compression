import torch
import numpy as np
import pandas as pd
from src.models.model_utils import apply_config
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
    
    # Convert params to config (bits and prune_amount per layer)
    config = {"layers": []}
    for i, group in enumerate(layer_groups):
        bits = int(max(4, min(16, params[i * 2])))  # Wider range for experimentation
        prune_amount = float(max(0.0, min(0.5, params[i * 2 + 1])))  # Adjusted range
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
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0,
                min_length=input_ids_batch.shape[1] + 1,
            )
            generated_ids_batch = gen_output
            
        total_robustness = 0.0
        falsification_count = 0
        sample_count = len(dataset_subset)
        
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
            
            if ground_truth_ids and generated_ids:
                gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                gt_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(f"Sample {i}: Input={dataset_subset[i]['input']}, Ground Truth='{gt_text}', Generated='{gen_text}'")
            
            robustness, falsified = monitor_stl_signals(base_signals, comp_signals_batch, specs, ground_truth_ids, input_len, generated_ids)
            for k in robustness:
                if not isinstance(robustness[k], (int, float)) or np.isnan(robustness[k]):
                    robustness[k] = 0.0
            total_robustness += sum(robustness.values())
            falsification_count += any(falsified.values())
            print(f"Iteration {iteration}: Sample {i}, Robustness={robustness}, Falsified={falsified}")
        
        compressed_size = sum(p.numel() * (config["layers"][i]["bits"] / 8) for i, p in enumerate(comp_model.parameters()))
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        objective = total_robustness / sample_count - compression_ratio  # Minimize robustness deviation, maximize compression
        
        print(f"Iteration {iteration}: Objective={objective:.4f}, Falsified Samples={falsification_count}")
        clear_gpu_memory()
        return objective
    
    except Exception as e:
        print(f"Iteration {iteration}: Error - {str(e)}")
        return float('inf')  # Return large value on failure

def run_optimization(base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir="./saved_models/", model_name="unknown"):
    from bayes_opt import BayesianOptimization
    
    pbounds = {f'p{i}': (4, 16) if i % 2 == 0 else (0, 0.5) for i in range(len(layer_groups) * 2)}  # bits: 4-16, prune: 0-0.5
    optimizer = BayesianOptimization(
        f=lambda **params: objective_function([params[f'p{i}'] for i in range(len(layer_groups) * 2)], base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir, model_name),
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    logs = pd.DataFrame(columns=['iteration', 'params', 'objective', 'falsified_samples', 'robustness'])
    
    def on_step(opt):
        global best_objective, best_params
        params = list(opt.res[-1]['params'].values())
        objective = opt.res[-1]['target']
        robustness, falsified_count = opt.space._cache[objective]['robustness'], opt.space._cache[objective]['falsified_count']
        logs.loc[len(logs)] = [iteration, params, objective, falsified_count, robustness]
        print(f"Params (first 6): {params[:6]}..., Falsified Samples: {falsified_count}, Robustness: {robustness}")
        if objective < best_objective:
            best_objective = objective
            best_params = params
    
    optimizer.maximize(init_points=5, n_iter=15)
    logs.to_csv(f"{save_dir}/{model_name}_optimization_log.csv", index=False)
    return best_params, logs