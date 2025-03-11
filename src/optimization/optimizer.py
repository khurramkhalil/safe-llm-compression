import torch
from scipy.optimize import differential_evolution
import numpy as np
import os
from src.models.model_utils import reset_model, apply_config, forward_with_signals_batched, log_quantized_size
from src.stl.stl_monitoring import monitor_stl_signals
from src.utils.common_utils import clear_gpu_memory

# Global variables for tracking optimization state
iteration = 0
best_objective = float('inf')
best_params = None
prev_params = None
prev_objective = float('inf')
falsified_configs = []

def objective_function(params, base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir="./saved_models/"):
    """Compute the objective value for a given set of optimization parameters."""
    global iteration, best_objective, best_params, prev_params, prev_objective, falsified_configs
    iteration += 1
    config = {"layers": []}
    for i, group in enumerate(layer_groups):
        bits = int(max(12, min(16, params[i * 2])))
        prune_amount = float(max(0, min(0.1, params[i * 2 + 1])))  # Convert np.float64 to float
        config["layers"].append({"pattern": group["pattern"], "bits": bits, "prune": prune_amount})
    
    sample_bits = [layer["bits"] for layer in config["layers"][:3]]
    sample_prune = [layer["prune"] for layer in config["layers"][:3]]
    print(f"Iteration {iteration}: Sampled Bits (first 3 layers): {sample_bits}, Pruning Ratios: {sample_prune}")
    
    try:
        # Configure model on CPU first
        comp_model = reset_model(base_model).to('cpu')
        comp_model = apply_config(comp_model, config)
        print(f"Iteration {iteration}: Model configured on CPU")
        
        # Move to GPU only for computation
        comp_model = comp_model.cuda()
        batch_size = 1
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(
            [tokenizer(seq["input"], return_tensors="pt", truncation=True, max_length=50).input_ids[0] for seq in dataset_subset],
            batch_first=True, padding_value=tokenizer.pad_token_id
        ).cuda()
        print(f"Iteration {iteration}: Input batch prepared with batch_size={batch_size}")
        
        comp_signals = forward_with_signals_batched(comp_model, input_ids_batch)
        print(f"Iteration {iteration}: Forward pass completed")
        
        total_robustness = 0.0
        falsification_count = 0
        robustness_sum = {k: 0.0 for k in ['seq_coh', 'long_range', 'ctx_cons', 'fact_acc']}
        sample_count = 0
        
        for i in range(len(dataset_subset)):
            try:
                base_signals = base_signals_cache[i]
                comp_signals_batch = {
                    "probs": comp_signals["probs"][i:i+1],
                    "attention_matrices": {k: v[i:i+1] for k, v in comp_signals["attention_matrices"].items()},
                    "hidden_states": comp_signals["hidden_states"][i:i+1]
                }
                ground_truth_ids = ground_truth_cache[i]
                input_len = dataset_subset[i]["input_len"]
                generated_ids = None
                if ground_truth_ids:
                    with torch.no_grad():
                        gen_output = comp_model.generate(
                            input_ids_batch[i:i+1],
                            max_new_tokens=len(ground_truth_ids),
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=None,
                            do_sample=False,
                            num_beams=1,  # Reduce beams to lower memory
                            return_dict_in_generate=True,
                            output_scores=False,  # Skip scores to save memory
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.2
                        )
                        generated_ids = gen_output.sequences[0, input_len:].tolist()
                        if len(generated_ids) < len(ground_truth_ids):
                            generated_ids.extend([tokenizer.pad_token_id] * (len(ground_truth_ids) - len(generated_ids)))
                        elif len(generated_ids) > len(ground_truth_ids):
                            generated_ids = generated_ids[:len(ground_truth_ids)]
                        gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        gt_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
                        print(f"Sample {i}: Input={dataset_subset[i]['input']}, Ground Truth='{gt_text}', Generated='{gen_text}'")
                robustness, falsified = monitor_stl_signals(base_signals, comp_signals_batch, specs, ground_truth_ids, input_len, generated_ids)
                total_robustness += sum(robustness.values())
                falsification_count += any(falsified.values())
                for k in robustness_sum:
                    robustness_sum[k] += robustness[k]
                sample_count += 1
                print(f"Iteration {iteration}: Sample {i} processed, Robustness={robustness}, Falsified={falsified}")
                
                # Clear intermediate tensors
                if 'gen_output' in locals():
                    del gen_output
            except Exception as e:
                print(f"Iteration {iteration}: Sample {i} failed - {str(e)}")
                robustness = {'seq_coh': 0.0, 'long_range': 0.0, 'ctx_cons': 0.0, 'fact_acc': 0.0}
                falsified = {'seq_coh': False, 'long_range': False, 'ctx_cons': False, 'fact_acc': False}
                total_robustness += sum(robustness.values())
                falsification_count += any(falsified.values())
                for k in robustness_sum:
                    robustness_sum[k] += robustness[k]
                sample_count += 1
            clear_gpu_memory()
        
        # Compute size and objective
        layer_bits = {l["pattern"]: l["bits"] for l in config["layers"]}
        size_mb = log_quantized_size(comp_model, layer_bits)
        compression_ratio = size_mb / original_size
        objective = -total_robustness + 1.0 * size_mb
        
        robustness_avg = {k: v / sample_count for k, v in robustness_sum.items()}
        
        # Save best model
        if objective < best_objective:
            if best_params is not None:
                prev_model = reset_model(base_model).to('cpu')
                prev_config = {"layers": [{"pattern": group["pattern"], "bits": int(max(12, min(16, best_params[j * 2]))), 
                                           "prune": float(max(0, min(0.1, best_params[j * 2 + 1])))} 
                                          for j, group in enumerate(layer_groups)]}
                prev_model = apply_config(prev_model, prev_config)
                prev_model.save_pretrained(os.path.join(save_dir, f"prev_best_model_iter_{iteration-1}"))
                del prev_model
            
            comp_model.to('cpu')  # Move to CPU before saving
            comp_model.save_pretrained(os.path.join(save_dir, f"best_model_iter_{iteration}"))
            prev_params = best_params
            prev_objective = best_objective
            best_objective = objective
            best_params = params.copy()
            print(f"Iteration {iteration}: New Best! Saved as best_model_iter_{iteration}")
        else:
            print(f"Iteration {iteration}: No improvement over best ({best_objective:.4f})")
        
        if falsification_count > 0:
            falsified_configs.append((params.copy(), falsification_count, robustness.copy()))
        
        print(f"Iteration {iteration}: Falsified {falsification_count}/{len(dataset_subset)} samples")
        print(f"Objective = {objective:.4f}, Total Robustness = {total_robustness:.4f}, "
              f"Seq_Coh = {robustness_avg['seq_coh']:.4f}, Long_Range = {robustness_avg['long_range']:.4f}, "
              f"Ctx_Cons = {robustness_avg['ctx_cons']:.4f}, Fact_Acc = {robustness_avg['fact_acc']:.4f}")
        print(f"Size = {size_mb:.2f} MB, Compression Ratio = {compression_ratio:.2%}")
        print(f"Iteration {iteration}: Objective computation completed")
        
        # Cleanup
        del comp_model, comp_signals, input_ids_batch
        clear_gpu_memory()
        return objective
    
    except Exception as e:
        print(f"Iteration {iteration}: Fatal error - {str(e)}")
        if 'comp_model' in locals():
            del comp_model
        clear_gpu_memory()
        raise

def run_optimization(base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir="./saved_models/", bounds=None):
    """Run differential evolution optimization to find the best compression configuration."""
    global iteration, best_objective, best_params, prev_params, prev_objective, falsified_configs
    
    iteration = 0
    best_objective = float('inf')
    best_params = None
    prev_params = None
    prev_objective = float('inf')
    falsified_configs = []
    
    if bounds is None:
        bounds = [(12, 16), (0, 0.1)] * len(layer_groups)
    
    result = differential_evolution(
        objective_function,
        bounds,
        args=(base_model, dataset_subset, tokenizer, layer_groups, base_signals_cache, ground_truth_cache, specs, original_size, save_dir),
        strategy='rand1bin',
        popsize=5,
        maxiter=2,
        tol=1e-3,
        disp=True,
        workers=1
    )
    
    print("\nOptimization Complete:")
    print(f"Best Objective: {best_objective:.4f}")
    print(f"Best Params (first 6): {best_params[:6] if best_params is not None else 'None'}...")
    print("\nFalsified Configurations:")
    for params, count, robustness in falsified_configs:
        print(f"Params (first 6): {params[:6]}..., Falsified Samples: {count}, Robustness: {robustness}")
    
    clear_gpu_memory()