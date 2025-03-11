import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi

models = ["llama_3.2_3b", "deepseek_7b", "gemma_2_2b", "phi_3.5"]
results_dir = "../results"

# Table 1: Overall Performance
data = []
for model in models:
    summary = pd.read_csv(f"{results_dir}/{model}_summary.csv")
    baseline = pd.read_csv(f"{results_dir}/{model}_baseline.csv")
    logs = pd.read_csv(f"{results_dir}/{model}_logs.csv")
    data.append([
        model.replace('_', ' ').title(),
        summary['original_size_mb'][0],
        summary['best_size_mb'][0],
        summary['best_compression_ratio'][0] * 100,
        baseline.mean().mean(),
        logs[['seq_coh', 'long_range', 'ctx_cons', 'fact_acc']].iloc[-1].mean(),
        logs['falsified'].iloc[-1],
        summary['runtime_s'][0]
    ])
df_table1 = pd.DataFrame(data, columns=["Model Name", "Original Size (MB)", "Compressed Size (MB)", "Compression Ratio (%)", 
                                        "Avg. Robustness (Baseline)", "Avg. Robustness (Compressed)", "Falsified Samples", "Runtime (s)"])
df_table1.to_csv(f"{results_dir}/table1_overall_performance.csv", index=False)

# Table 2: STL Robustness Breakdown
data = []
for model in models:
    baseline = pd.read_csv(f"{results_dir}/{model}_baseline.csv").iloc[0]
    logs = pd.read_csv(f"{results_dir}/{model}_logs.csv").iloc[-1]
    data.append([model.replace('_', ' ').title(),
                 f"{baseline['seq_coh']:.2f} / {logs['seq_coh']:.2f}",
                 f"{baseline['long_range']:.2f} / {logs['long_range']:.2f}",
                 f"{baseline['ctx_cons']:.2f} / {logs['ctx_cons']:.2f}",
                 f"{baseline['fact_acc']:.2f} / {logs['fact_acc']:.2f}"])
df_table2 = pd.DataFrame(data, columns=["Model Name", "Seq_Coh (Base/Comp)", "Long_Range (Base/Comp)", "Ctx_Cons (Base/Comp)", "Fact_Acc (Base/Comp)"])
df_table2.to_csv(f"{results_dir}/table2_robustness_breakdown.csv", index=False)

# Table 3: Best Configuration
data = []
for model in models:
    logs = pd.read_csv(f"{results_dir}/{model}_logs.csv").iloc[-1]
    data.append([model.replace('_', ' ').title(),
                 f"{int(logs['bits_1'])} / {logs['prune_1']:.2f}",
                 f"{int(logs['bits_2'])} / {logs['prune_2']:.2f}",
                 f"{int(logs['bits_3'])} / {logs['prune_3']:.2f}",
                 logs['iteration']])
df_table3 = pd.DataFrame(data, columns=["Model Name", "Layer 1 Bits/Prune", "Layer 2 Bits/Prune", "Layer 3 Bits/Prune", "Total Iterations"])
df_table3.to_csv(f"{results_dir}/table3_best_config.csv", index=False)

# Figure 1: Compression Ratio Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(df_table1["Model Name"], df_table1["Compression Ratio (%)"], color='skyblue')
plt.axhline(y=30, color='r', linestyle='--', label='Target 30%')
plt.xlabel("Model")
plt.ylabel("Compression Ratio (%)")
plt.title("Compression Ratio Across Models")
plt.legend()
plt.xticks(rotation=15)
plt.savefig(f"{results_dir}/fig1_compression_ratio.png")
plt.close()

# Figure 2: Robustness Spider Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ['Seq_Coh', 'Long_Range', 'Ctx_Cons', 'Fact_Acc']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
for i, model in enumerate(models):
    baseline = pd.read_csv(f"{results_dir}/{model}_baseline.csv").iloc[0]
    comp = pd.read_csv(f"{results_dir}/{model}_logs.csv").iloc[-1]
    values_base = [baseline['seq_coh'], baseline['long_range'], baseline['ctx_cons'], baseline['fact_acc']] + [baseline['seq_coh']]
    values_comp = [comp['seq_coh'], comp['long_range'], comp['ctx_cons'], comp['fact_acc']] + [comp['seq_coh']]
    ax.plot(angles, values_base, linewidth=2, linestyle='solid', label=f"{model} Baseline")
    ax.plot(angles, values_comp, linewidth=2, linestyle='dashed', label=f"{model} Compressed")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title("Robustness Comparison")
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.savefig(f"{results_dir}/fig2_robustness_spider.png")
plt.close()

# Figure 3: Objective Value Over Iterations
plt.figure(figsize=(10, 6))
for model in models:
    df = pd.read_csv(f"{results_dir}/{model}_logs.csv")
    plt.plot(df["iteration"], df["objective"], label=model.replace('_', ' ').title())
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Objective Value Trends")
plt.legend()
plt.savefig(f"{results_dir}/fig3_objective_trend.png")
plt.close()

# Figure 4: Size vs. Robustness Scatter
plt.figure(figsize=(10, 6))
for model in models:
    baseline = pd.read_csv(f"{results_dir}/{model}_baseline.csv").mean().mean()
    summary = pd.read_csv(f"{results_dir}/{model}_summary.csv")
    comp = pd.read_csv(f"{results_dir}/{model}_logs.csv").iloc[-1][['seq_coh', 'long_range', 'ctx_cons', 'fact_acc']].mean()
    plt.scatter(summary['original_size_mb'], baseline, marker='o', label=f"{model} Baseline")
    plt.scatter(summary['best_size_mb'], comp, marker='x', label=f"{model} Compressed")
    plt.plot([summary['original_size_mb'], summary['best_size_mb']], [baseline, comp], linestyle='--')
plt.xlabel("Size (MB)")
plt.ylabel("Average Robustness")
plt.title("Size vs. Robustness Trade-off")
plt.legend()
plt.savefig(f"{results_dir}/fig4_size_vs_robustness.png")
plt.close()

# Figure 5: Runtime Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(df_table1["Model Name"], df_table1["Runtime (s)"], color='lightgreen')
plt.xlabel("Model")
plt.ylabel("Runtime (s)")
plt.title("Optimization Runtime")
plt.xticks(rotation=15)
plt.savefig(f"{results_dir}/fig5_runtime.png")
plt.close()

print("Tables and figures generated in results/")

# Figure 6: Pareto-Front (Energy Gains vs. STL Robustness)
plt.figure(figsize=(10, 6))
for model in models:
    df = pd.read_csv(f"{results_dir}/{model}_logs.csv")
    avg_robustness = df[['seq_coh', 'long_range', 'ctx_cons', 'fact_acc']].mean(axis=1)
    plt.scatter(df["energy_gains"], avg_robustness, label=model.replace('_', ' ').title(), alpha=0.6)
plt.xlabel("Energy Gains (%)")
plt.ylabel("Average STL Robustness")
plt.title("Pareto-Front: Energy Gains vs. STL Robustness")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{results_dir}/fig6_pareto_front.png")
plt.close()

print("Tables and figures generated in results/")