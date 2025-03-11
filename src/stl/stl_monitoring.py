import torch
import torch.nn.functional as F
from rtamt import StlDiscreteTimeSpecification

def define_stl_specifications(k=5, epsilon=0.1, delta=0.2, gamma=0.4, tau=0.5):
    """Define STL specifications for model behavior monitoring."""
    # Sequence Coherence: JSD <= epsilon for first k tokens
    seq_coh_spec = StlDiscreteTimeSpecification()
    seq_coh_spec.declare_var('jsd', 'float')
    seq_coh_spec.spec = f'always[0:{k}] (jsd <= {epsilon})'
    seq_coh_spec.parse()

    # Long-Range Dependency: Cosine similarity of attention >= delta
    long_range_spec = StlDiscreteTimeSpecification()
    long_range_spec.declare_var('cos_attn', 'float')
    long_range_spec.spec = f'always (cos_attn >= {delta})'
    long_range_spec.parse()

    # Contextual Consistency: Cosine similarity of hidden states >= gamma
    ctx_cons_spec = StlDiscreteTimeSpecification()
    ctx_cons_spec.declare_var('cos_hidden', 'float')
    ctx_cons_spec.spec = f'always (cos_hidden >= {gamma})'
    ctx_cons_spec.parse()

    # Factual Accuracy: Probability ratio >= tau
    fact_acc_spec = StlDiscreteTimeSpecification()
    fact_acc_spec.declare_var('prob_ratio', 'float')
    fact_acc_spec.spec = f'always (prob_ratio >= {tau})'
    fact_acc_spec.parse()

    return seq_coh_spec, long_range_spec, ctx_cons_spec, fact_acc_spec

def compute_signals(base_signals, comp_signals, ground_truth_ids=None, input_len=None, generated_ids=None):
    """Compute signals for STL monitoring from base and compressed model outputs."""
    # Use first attention layer, squeeze batch dimension
    attn_base = list(base_signals["attention_matrices"].values())[0].squeeze(0) if base_signals["attention_matrices"] else None
    attn_comp = list(comp_signals["attention_matrices"].values())[0].squeeze(0) if comp_signals["attention_matrices"] else None
    seq_len = attn_base.size(0) if attn_base is not None else base_signals["probs"].size(1)
    signals = {}
    time_values = list(range(seq_len))

    # Debug shapes
    print(f"Debug: probs shape = {base_signals['probs'].shape}")
    print(f"Debug: hidden_states len = {len(base_signals['hidden_states'])}, last layer shape = {base_signals['hidden_states'][-1].shape}")

    # Sequence Coherence: Jensen-Shannon Divergence
    jsd_values = []
    probs_base = base_signals["probs"].squeeze(0)  # [seq_len, vocab_size]
    probs_comp = comp_signals["probs"].squeeze(0)  # [seq_len, vocab_size]
    for t in range(seq_len):
        p_base = probs_base[t]
        p_comp = probs_comp[t]
        p_sorted, _ = torch.sort(p_base, descending=True)
        q_sorted, _ = torch.sort(p_comp, descending=True)
        cumsum_p = torch.cumsum(p_sorted, dim=-1)
        k_top = (cumsum_p < 0.9).sum() + 1
        p_top = p_sorted[:k_top] / p_sorted[:k_top].sum()
        q_top = q_sorted[:k_top] / q_sorted[:k_top].sum()
        m = 0.5 * (p_top + q_top)
        jsd = 0.5 * (F.kl_div(torch.log(p_top + 1e-10), m, reduction='sum') +
                     F.kl_div(torch.log(q_top + 1e-10), m, reduction='sum'))
        jsd_values.append(jsd.item())
    signals["seq_coh"] = {'time': time_values, 'jsd': jsd_values}

    # Long-Range Dependency: Cosine similarity of attention matrices
    cos_attn_values = []
    if attn_base is not None and attn_comp is not None:
        for t in range(seq_len):
            if attn_base[t].sum() != 0 and attn_comp[t].sum() != 0:
                similarities = torch.cosine_similarity(attn_base[t], attn_comp[t], dim=-1)
                cos_value = similarities.mean().item()
            else:
                cos_value = 1.0  # Default if attention is zero
            cos_attn_values.append(cos_value)
    else:
        cos_attn_values = [1.0] * seq_len
    signals["long_range"] = {'time': time_values, 'cos_attn': cos_attn_values}

    # Contextual Consistency: Cosine similarity of hidden states
    h_base = base_signals["hidden_states"][-1].squeeze(0)  # Last layer, [seq_len, hidden_dim]
    h_comp = comp_signals["hidden_states"][-1].squeeze(0)  # Last layer, [seq_len, hidden_dim]
    print(f"Debug: h_base shape after squeeze = {h_base.shape}")
    cos_hidden_values = []
    for t in range(seq_len):
        h_base_t = h_base[t]  # [hidden_dim]
        h_comp_t = h_comp[t]  # [hidden_dim]
        print(f"Debug: t={t}, h_base[t] shape = {h_base_t.shape}, h_comp[t] shape = {h_comp_t.shape}")
        cos_value = torch.cosine_similarity(h_base_t.unsqueeze(0), h_comp_t.unsqueeze(0), dim=-1).item()  # Add dim to make [1, hidden_dim]
        cos_hidden_values.append(cos_value)
    signals["ctx_cons"] = {'time': time_values, 'cos_hidden': cos_hidden_values}

    # Factual Accuracy: Probability ratio for ground truth tokens
    if ground_truth_ids is not None and input_len is not None and generated_ids is not None:
        prob_ratio_values = [1.0] * seq_len
        for i, t in enumerate(range(input_len, min(input_len + len(ground_truth_ids), seq_len))):
            correct_id = ground_truth_ids[i]
            gen_id = generated_ids[i]
            p_base_correct = probs_base[t, correct_id].item()
            p_comp_correct = probs_comp[t, correct_id].item()
            ratio = p_comp_correct / p_base_correct if p_base_correct > 0 else 1.0 if gen_id == correct_id else 0.0
            prob_ratio_values[t] = ratio
        signals["fact_acc"] = {'time': time_values, 'prob_ratio': prob_ratio_values}

    return signals

def monitor_stl_signals(base_signals, comp_signals, specs, ground_truth_ids=None, input_len=None, generated_ids=None):
    """Evaluate STL specifications on computed signals and return robustness and falsification."""
    seq_coh_spec, long_range_spec, ctx_cons_spec, fact_acc_spec = specs
    signals = compute_signals(base_signals, comp_signals, ground_truth_ids, input_len, generated_ids)
    
    robustness_scores = {}
    falsified = {}

    # Sequence Coherence
    robustness_trace = seq_coh_spec.evaluate(signals["seq_coh"])
    robustness_scores["seq_coh"] = min(r for _, r in robustness_trace)
    falsified["seq_coh"] = robustness_scores["seq_coh"] < 0

    # Long-Range Dependency
    robustness_trace = long_range_spec.evaluate(signals["long_range"])
    robustness_scores["long_range"] = min(r for _, r in robustness_trace)
    falsified["long_range"] = robustness_scores["long_range"] < 0

    # Contextual Consistency
    robustness_trace = ctx_cons_spec.evaluate(signals["ctx_cons"])
    robustness_scores["ctx_cons"] = min(r for _, r in robustness_trace)
    falsified["ctx_cons"] = robustness_scores["ctx_cons"] < 0

    # Factual Accuracy (only if ground truth is provided)
    if "fact_acc" in signals:
        robustness_trace = fact_acc_spec.evaluate(signals["fact_acc"])
        robustness_scores["fact_acc"] = min(r for _, r in robustness_trace)
        falsified["fact_acc"] = robustness_scores["fact_acc"] < 0
    else:
        robustness_scores["fact_acc"] = 0.0
        falsified["fact_acc"] = False

    return robustness_scores, falsified