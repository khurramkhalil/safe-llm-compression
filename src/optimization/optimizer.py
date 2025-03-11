import torch
from scipy.optimize import differential_evolution
import numpy as np
import os
import csv
import time
import pandas as pd
from src.models.model_utils import apply_config, forward_with_signals_batched, log_quantized_size
from src.stl.stl_monitoring import monitor_stl_signals
from src.utils.common_utils import clear_gpu_memory

iteration = 0
best_objective = float('inf')
best_params = None
falsified_configs = []

def objective_function(params, base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir="./saved_models/", model_name="unknown"):
    global iteration, best_objective, best_params, falsified_configs
    iteration += 1
    start_iter_time = time.time()
    
    config = {"layers": []}  # Fixed syntax error: removed duplicate 'config'
    for i, group in enumerate(layer_groups):
        bits = int(max(14, min(16, params[i * 2])))
        prune_amount = float(max(0, min(0.05, params[i * 2 + 1])))
        config["layers"].append({"pattern": group["pattern"], "bits": bits, "prune": prune_amount})
    
    sample_bits = [layer["bits"] for layer in config["layers"][:3]]
    sample_prune = [layer["prune"] for layer in config["layers"][:3]]
    print(f"Iteration {iteration}: Sampled Bits (first 3 layers): {sample_bits}, Pruning Ratios: {sample_prune}")
    
    try:
        comp_model = apply_config(base_model, config)
        print(f"Iteration {iteration}: Model configured on GPU")
        
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(
            [tokenizer(seq["input"], return_tensors="pt", truncation=True, max_length=50, padding='max_length').input_ids[0] for seq in dataset_subset],
            batch_first=True, padding_value=tokenizer.pad_token_id
        ).cuda()
        print(f"Iteration {iteration}: Input batch prepared with batch_size={input_ids_batch.shape[0]}")
        
        comp_signals = forward_with_signals_batched(comp_model, input_ids_batch)
        print(f"Iteration {iteration}: Forward pass completed")
        
        max_new_tokens = max([len(gt) for gt in ground_truth_cache if gt is not None])
        with torch.no_grad():
            gen_output = comp_model.generate(
                input_ids_batch,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.0
            )
            generated_ids_batch = gen_output.sequences
            del gen_output
        
        total_robustness = 0.0
        falsification_count = 0
        robustness_sum = {k: 0.0 for k in ['seq_coh', 'long_range', 'ctx_cons', 'fact_acc']}
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
                gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                gt_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
                print(f"Sample {i}: Input={dataset_subset[i]['input']}, Ground Truth='{gt_text}', Generated='{gen_text}'")
            
            robustness, falsified = monitor_stl_signals(base_signals, comp_signals_batch, specs, ground_truth_ids, input_len, generated_ids)
            for k in robustness:
                if not isinstance(robustness[k], (int, float)) or np.isnan(robustness[k]):
                    robustness[k] = 0.0
            total_robustness += sum(robustness.values())
            falsification_count += any(falsified.values())
            for k in robustness_sum:
                robustness_sum[k] += robustness[k]
            print(f"Iteration {iteration}: Sample {i} processed, Robustness={robustness}, Falsified={falsified}")
        
        layer_bits = {l["pattern"]: l["bits"] for l in config["layers"]}
        size_mb = log_quantized_size(comp_model, layer_bits)
        compression_ratio = size_mb / original_size
        energy_gains = (original_size - size_mb) / original_size * 100
        objective = -total_robustness + 1.0 * size_mb
        
        robustness_avg = {k: v / sample_count for k, v in robustness_sum.items()}
        
        os.makedirs("results", exist_ok=True)
        log_file = f"results/{model_name}_logs.csv"
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if iteration == 1:
                writer.writerow(["iteration", "objective", "size_mb", "compression_ratio", "seq_coh", "long_range", "ctx_cons", "fact_acc", "falsified", "bits_1", "prune_1", "bits_2", "prune_2", "bits_3", "prune_3", "energy_gains"])
            writer.writerow([iteration, objective, size_mb, compression_ratio, robustness_avg['seq_coh'], robustness_avg['long_range'], robustness_avg['ctx_cons'], robustness_avg['fact_acc'], falsification_count, 
                            sample_bits[0], sample_prune[0], sample_bits[1], sample_prune[1], sample_bits[2], sample_prune[2], energy_gains])
        
        if objective < best_objective and not np.isnan(objective):
            comp_model.save_pretrained(os.path.join(save_dir, f"best_model_iter_{iteration}"))
            best_objective = objective
            best_params = params.copy()
            print(f"Iteration {iteration}: New Best! Saved as best_model_iter_{iteration}")
        else:
            print(f"Iteration {iteration}: No improvement over best ({best_objective:.4f})")
        
        if falsification_count > 0:
            falsified_configs.append((params.copy(), falsification_count, robustness.copy()))
        
        print(f"Iteration {iteration}: Falsified {falsification_count}/{sample_count} samples")
        print(f"Objective = {objective:.4f}, Total Robustness = {total_robustness:.4f}, "
              f"Seq_Coh = {robustness_avg['seq_coh']:.4f}, Long_Range = {robustness_avg['long_range']:.4f}, "
              f"Ctx_Cons = {robustness_avg['ctx_cons']:.4f}, Fact_Acc = {robustness_avg['fact_acc']:.4f}")
        print(f"Size = {size_mb:.2f} MB, Compression Ratio = {compression_ratio:.2%}")
        
        iter_time = time.time() - start_iter_time
        print(f"Iteration {iteration}: Objective computation completed in {iter_time:.2f} seconds")
        
        del comp_signals, input_ids_batch, generated_ids_batch
        return objective
    
    except Exception as e:
        print(f"Iteration {iteration}: Fatal error - {str(e)}")
        clear_gpu_memory()
        raise

def run_optimization(base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir="./saved_models/", bounds=None, model_name="unknown"):
    global iteration, best_objective, best_params, falsified_configs
    
    iteration = 0
    best_objective = float('inf')
    best_params = None
    falsified_configs = []
    
    if bounds is None:
        bounds = [(14, 16), (0, 0.05)] * len(layer_groups)
    
    start_time = time.time()
    base_model.cuda()
    result = differential_evolution(
        objective_function,
        bounds,
        args=(base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir, model_name),
        strategy='rand1bin',
        popsize=5,
        maxiter=2,
        tol=1e-3,
        disp=True,
        workers=1
    )
    runtime = time.time() - start_time
    
    print("\nOptimization Complete:")
    print(f"Best Objective: {best_objective:.4f}")
    print(f"Best Params (first 6): {best_params[:6] if best_params is not None else 'None'}...")
    print(f"Runtime: {runtime:.2f} seconds")
    print("\nFalsified Configurations:")
    for params, count, robustness in falsified_configs:
        print(f"Params (first 6): {params[:6]}..., Falsified Samples: {count}, Robustness: {robustness}")
    
    log_file = f"results/{model_name}_logs.csv"
    if os.path.exists(log_file):
        logs = pd.read_csv(log_file)
        best_row = logs.loc[logs['objective'].idxmin()]
        best_size_mb = best_row['size_mb']
        best_compression_ratio = best_row['compression_ratio']
        best_energy_gains = best_row['energy_gains']
        os.makedirs("results", exist_ok=True)
        with open(f"results/{model_name}_summary.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["original_size_mb", "best_size_mb", "best_compression_ratio", "runtime_s", "best_energy_gains"])
            writer.writerow([original_size, best_size_mb, best_compression_ratio, runtime, best_energy_gains])
    
    clear_gpu_memory()

if __name__ == "__main__":
    # This is typically not run directly, but included for completeness
    pass