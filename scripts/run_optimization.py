import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import yaml
import csv
import time
import gc
from src.models.model_utils import load_base_model, forward_with_signals_batched, get_layer_groups
from src.data.dataset_utils import load_and_preprocess_data
from src.stl.stl_monitoring import define_stl_specifications, monitor_stl_signals
from src.optimization.optimizer import run_optimization
from src.utils.common_utils import clear_gpu_memory

def load_config(config_path="config/config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function to run the LLM compression optimization."""
    config = load_config()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    save_dir = config['model']['save_dir']
    model_name = config['model']['name'].replace('/', '_')

    start_time = time.time()

    print("Loading base model...")
    base_model, tokenizer, original_size = load_base_model(
        model_name=config['model']['name'],
        save_dir=save_dir
    )

    print("Preprocessing dataset...")
    dataset_subset, ground_truth_cache = load_and_preprocess_data(
        tokenizer=tokenizer,
        num_samples=config['dataset']['num_samples']
    )
    preprocess_time = time.time() - start_time

    print("Computing base signals...")
    batch_size = config['misc']['batch_size']
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(
        [tokenizer(seq["input"], return_tensors="pt", truncation=True, max_length=config['dataset']['max_length']).input_ids[0] for seq in dataset_subset],
        batch_first=True, padding_value=tokenizer.pad_token_id
    ).cuda()
    base_signals_full_batch = forward_with_signals_batched(base_model, input_ids_batch)
    # Keep base_signals_cache on GPU until optimization
    base_signals_cache = [
        {
            "probs": base_signals_full_batch["probs"][i:i+1],
            "attention_matrices": {k: v[i:i+1] for k, v in base_signals_full_batch["attention_matrices"].items()},
            "hidden_states": base_signals_full_batch["hidden_states"][i:i+1]
        }
        for i in range(len(dataset_subset))
    ]
    print("Base signals computed for all samples")
    del base_signals_full_batch
    gc.collect()  # Force garbage collection

    print("Performing baseline evaluation...")
    specs = define_stl_specifications(
        k=config['stl']['k'],
        epsilon=config['stl']['epsilon'],
        delta=config['stl']['delta'],
        gamma=config['stl']['gamma'],
        tau=config['stl']['tau']
    )
    base_signals = base_signals_cache[0]
    ground_truth_ids = ground_truth_cache[0]
    input_len = dataset_subset[0]["input_len"]
    generated_ids = None
    if ground_truth_ids:
        input_ids_batch = tokenizer(dataset_subset[0]["input"], return_tensors="pt", truncation=True, max_length=config['dataset']['max_length']).input_ids.cuda()
        with torch.no_grad():
            gen_output = base_model.generate(
                input_ids_batch,
                max_new_tokens=len(ground_truth_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )
            generated_ids = gen_output.sequences[0, input_len:].tolist()
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            gt_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
            print(f"Baseline Sample 0: Input={dataset_subset[0]['input']}, Ground Truth='{gt_text}', Generated='{gen_text}'")
            del gen_output
        del input_ids_batch
        clear_gpu_memory()
    robustness, falsified = monitor_stl_signals(base_signals, base_signals, specs, ground_truth_ids, input_len, generated_ids)
    print("Uncompressed Robustness:", robustness, "Falsified:", falsified)
    print(f"Baseline: Seq_Coh={robustness['seq_coh']:.4f}, Long_Range={robustness['long_range']:.4f}, "
          f"Ctx_Cons={robustness['ctx_cons']:.4f}, Fact_Acc={robustness['fact_acc']:.4f}")

    # Log baseline
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_baseline.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["seq_coh", "long_range", "ctx_cons", "fact_acc"])
        writer.writerow([robustness['seq_coh'], robustness['long_range'], robustness['ctx_cons'], robustness['fact_acc']])

    print("Starting optimization...")
    layer_groups = get_layer_groups(base_model)
    bounds = [tuple(config['optimization']['bounds']['bits']), tuple(config['optimization']['bounds']['prune'])] * len(layer_groups)
    run_optimization(
        base_model=base_model,
        dataset_subset=dataset_subset,
        tokenizer=tokenizer,
        layer_groups=layer_groups,
        base_signals_cache=base_signals_cache,
        ground_truth_cache=ground_truth_cache,
        specs=specs,
        original_size=original_size,
        save_dir=save_dir,
        model_name=model_name
    )

    total_time = time.time() - start_time
    print(f"Total Runtime: {total_time:.2f} seconds, Preprocessing: {preprocess_time:.2f} seconds")
    
    del base_model, base_signals_cache  # Final cleanup
    clear_gpu_memory()
    gc.collect()
    print("Optimization completed successfully.")

if __name__ == "__main__":
    main()