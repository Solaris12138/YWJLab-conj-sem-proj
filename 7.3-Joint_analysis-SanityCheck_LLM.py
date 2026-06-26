import os
import mne
import pandas as pd
import numpy as np
import argparse
import torch
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils.utils import mm_to_inches
from utils.configs import TOPIC_SEM_SETTINGS
from utils.llm_features import extract_llm_features


# Global Parameters
SAVE_ROOT = "./data/derivatives/materials"
IMAGE_DIR = "./results/7.3-Sanity_check"
NUM_TOPICS = 7
CONDS = {
    "Sta" : [[0, 0], [1, 1]], # Based on "topic_sem_settings": [context_idx, target_idx]
    "DevSem" : [[0, 1], [1, 0]],
    "DevSynC2" : [[2, 1], [3, 0]],
    "DevSynC1" : [[2, 0], [3, 1]]
}
MODELS = [
    "Atom-7B",
    "Qwen3-0.6B-Base",
    "Qwen3-1.7B-Base",
    "Qwen3-4B-Base",
    "Qwen3-8B-Base",
    "Qwen3-14B-Base"
]
SEED = 1016

TICK_FONTSIZE = 6
AXIS_LABEL_FONTSIZE = 7
LEGEND_FONTSIZE = 6
PANEL_TITLE_FONTSIZE = 8
MODEL_LABEL_FONTSIZE = 6
STATUS_LABEL_FONTSIZE = 6


# Self-defined functions
def run_extraction():
    all_results = list()

    for model_name in MODELS:
        print(f"\n>>> Processing Model: {model_name}")
        model_path = os.path.join("./utils/huggingface", model_name)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                        device_map="auto",
                                                        torch_dtype=torch.float16,
                                                        quantization_config=quantization_config,
                                                        trust_remote_code=True,
                                                        attn_implementation="flash_attention_2")
            model = model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                    use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
    
        for topic_idx in range(1, NUM_TOPICS + 1):
            topic_key = f"topic{topic_idx}"
            contexts = TOPIC_SEM_SETTINGS[topic_key]["contexts"]
            targets = TOPIC_SEM_SETTINGS[topic_key]["targets"]

            for cond, combinations in CONDS.items():
                for comb_idx, (ctx_idx, tgt_idx) in enumerate(combinations):
                    sentence = contexts[ctx_idx] + targets[tgt_idx]
                    features = extract_llm_features(sentence=sentence, model=model, tokenizer=tokenizer)
                    if len(features[-1]["word"]) == 1:
                        # S(w) = [S(t1 | C) + S(t2 | C, t1)] / 2
                        surprisal = (features[-1]["surprisal"] + features[-2]["surprisal"]) / 2
                    else:
                        surprisal = features[-1]["surprisal"]
                    
                    all_results.append({
                        "Model": model_name,
                        "Topic": topic_key,
                        "Condition": cond,
                        "Comb_Idx": comb_idx,
                        "Sentence": sentence,
                        "Surprisal": surprisal
                    })
    
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    df = pd.DataFrame(all_results)
    save_path = os.path.join(SAVE_ROOT, "llm_surprisal.csv")
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")


def _safe_sem_ci(values):
    """
    Return mean and approximate 95% CI half-width.
    """
    values = pd.Series(values).dropna().astype(float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values.iloc[0], 0.0
    return values.mean(), 1.96 * values.sem(ddof=1)


def _zscore(s):
    """
    Parameters
    ----------
    s: pd.Series

    Returns
    -------
    pd.Series
    """
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def _condition_metadata(condition, comb_idx):
    """
    Recover context/target indices and a simple clause-relation label from CONDS.

    In TOPIC_SEM_SETTINGS:
    target_idx == 0 corresponds to the causal target,
    target_idx == 1 corresponds to the concessive/adversative target.
    """
    ctx_idx, target_idx = CONDS[condition][int(comb_idx)]
    clause_relation = "Causal" if target_idx == 0 else "Concessive"
    return ctx_idx, target_idx, clause_relation


def _prepare_surprisal_dataframe(df):
    """
    Add plotting metadata used by panels.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    meta = df.apply(
        lambda r: _condition_metadata(r["Condition"], r["Comb_Idx"]),
        axis=1,
        result_type="expand",
    )
    meta.columns = ["ContextIdx", "TargetIdx", "ClauseRelation"]
    df = pd.concat([df, meta], axis=1)
    df["Group"] = df["Condition"] + "_" + df["Comb_Idx"].astype(str)
    df["Surprisal_z_model"] = df.groupby("Model")["Surprisal"].transform(_zscore)
    return df


def _apply_axis_style(ax):
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE,
                   length=2.0, width=0.45, pad=1.5)
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.35)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(TICK_FONTSIZE)
        tick_label.set_fontweight("normal")
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)


def _legend_font_properties():
    return mpl.font_manager.FontProperties(size=LEGEND_FONTSIZE, weight="bold")


def _plot_panel_A(ax, df, condition_order, model_order, palette, excluded_models):
    """Panel A: model-wise validation before model exclusion."""
    rng = np.random.default_rng(SEED)

    group_width = len(condition_order) + 1.25
    x_lookup = {}
    all_x = []
    all_xticklabels = []

    y_min = df["Surprisal"].min()
    y_max = df["Surprisal"].max()
    y_span = y_max - y_min if y_max > y_min else 1

    for mi, model in enumerate(model_order):
        base = mi * group_width
        if model in excluded_models:
            ax.axvspan(base - 0.55, base + len(condition_order) - 0.45,
                       color="#F1B6B6", alpha=0.20, zorder=0)

        means_for_line = []
        xs_for_line = []
        for ci, cond in enumerate(condition_order):
            x = base + ci
            x_lookup[(model, cond)] = x
            all_x.append(x)
            all_xticklabels.append(cond)

            vals = df.loc[(df["Model"] == model) & (df["Condition"] == cond), "Surprisal"]
            jitter = rng.normal(0, 0.045, size=len(vals))
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=4,
                color=palette[cond],
                alpha=0.30,
                linewidths=0,
                zorder=2
            )

            mean, ci95 = _safe_sem_ci(vals)
            ax.errorbar(
                x,
                mean,
                yerr=ci95,
                fmt="o",
                markersize=2.8,
                color=palette[cond],
                markeredgecolor="white",
                markeredgewidth=0.45,
                elinewidth=0.5,
                capsize=1.2,
                zorder=4,
            )
            means_for_line.append(mean)
            xs_for_line.append(x)

        ax.plot(xs_for_line, means_for_line, color="#2F2F2F", linewidth=0.55, alpha=0.55, zorder=3)

        center = base + (len(condition_order) - 1) / 2
        ax.text(center, y_min - 0.1 * y_span, model, ha="center", va="top",
                fontsize=MODEL_LABEL_FONTSIZE, rotation=0, clip_on=False)
        tag = "Excluded" if model in excluded_models else ""
        tag_color = "#A34040" if model in excluded_models else "#5C5C5C"
        ax.text(center, y_max + 0.10 * y_span, tag, ha="center", va="bottom",
                fontsize=LEGEND_FONTSIZE, color=tag_color, clip_on=False)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(-0.8, (len(model_order) - 1) * group_width + len(condition_order) - 0.2)
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.18 * y_span)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_ylabel("Target-Word Surprisal (Nats)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", labelpad=3)
    ax.set_title("A  Model-Wise Validation Before Consensus Construction", loc="left",
                 fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold", pad=8)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.45)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=True)
    _apply_axis_style(ax)

    handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", markersize=3.2,
                         markerfacecolor=palette[c], markeredgecolor="white",
                         label=c)
        for c in condition_order
    ]
    ax.legend(handles=handles, frameon=False, prop=_legend_font_properties(), loc="upper left",
              bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0,
              handletextpad=0.35, labelspacing=0.35, borderpad=0.2)


def _plot_panel_B(ax, df, condition_order, palette, excluded_models):
    """Panel B: consensus surprisal after excluding failed validation models."""
    validated = df.loc[~df["Model"].isin(excluded_models)].copy()

    # Consensus is computed on model-wise z-scored surprisal so that models
    # contribute equally despite different absolute surprisal scales.
    consensus = (
        validated
        .groupby(["Topic", "Condition", "Comb_Idx", "ClauseRelation", "Group"], as_index=False)
        .agg(ConsensusSurprisal=("Surprisal_z_model", "mean"),
             n_models=("Model", "nunique"))
    )

    relation_order = ["Causal", "Concessive"]
    offsets = {"Causal": -0.17, "Concessive": 0.17}
    markers = {"Causal": "o", "Concessive": "s"}
    rng = np.random.default_rng(SEED)

    for ri, relation in enumerate(relation_order):
        xs_line = []
        ys_line = []
        for ci, cond in enumerate(condition_order):
            x = ci + offsets[relation]
            vals = consensus.loc[
                (consensus["Condition"] == cond) &
                (consensus["ClauseRelation"] == relation),
                "ConsensusSurprisal"
            ]
            jitter = rng.normal(0, 0.025, size=len(vals))
            face = palette[cond] if relation == "Causal" else "white"
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=7.0,
                marker=markers[relation],
                facecolor=face,
                edgecolor=palette[cond],
                linewidth=0.55,
                alpha=0.5,
                zorder=2,
            )
            mean, ci95 = _safe_sem_ci(vals)
            ax.errorbar(
                x,
                mean,
                yerr=ci95,
                fmt=markers[relation],
                markersize=3.2,
                markerfacecolor=face,
                markeredgecolor=palette[cond],
                markeredgewidth=0.65,
                ecolor=palette[cond],
                elinewidth=0.75,
                capsize=1.8,
                zorder=4,
            )
            xs_line.append(x)
            ys_line.append(mean)
        ax.plot(xs_line, ys_line, color="#4B4B4B", linewidth=0.55,
                alpha=0.45, linestyle="-" if relation == "Causal" else "--", zorder=1)

    ax.axhline(0, color="#BBBBBB", linewidth=0.55, linestyle="--", zorder=0)
    ax.set_xticks(range(len(condition_order)))
    ax.set_xticklabels([
        "Sta",
        "DevSynC2",
        "DevSynC1",
        "DevSem",
    ], fontsize=TICK_FONTSIZE)
    ax.set_ylabel("Consensus Surprisal\n(Z-Scored Within Models)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", labelpad=3)
    ax.set_title("B  Consensus Surprisal After Excluding the Failed-Validation Model", loc="left",
                 fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold", pad=8)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.45)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=True)
    _apply_axis_style(ax)

    relation_handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", markersize=3.2,
                         markerfacecolor="#4B4B4B", markeredgecolor="#4B4B4B",
                         label="Causal Relation"),
        mpl.lines.Line2D([0], [0], marker="s", linestyle="", markersize=3.2,
                         markerfacecolor="white", markeredgecolor="#4B4B4B",
                         label="Concessive Relation"),
    ]
    ax.legend(handles=relation_handles, frameon=False, prop=_legend_font_properties(),
              loc="upper left", bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0,
              handletextpad=0.35, labelspacing=0.35, borderpad=0.2)


def run_visualization():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    file_path = os.path.join(SAVE_ROOT, "llm_surprisal.csv")
    if not os.path.exists(file_path):
        print("Data file not found. Please run with --prepare first.")
        return

    df = pd.read_csv(file_path)
    df = _prepare_surprisal_dataframe(df)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONTSIZE
    plt.rcParams["xtick.labelsize"] = TICK_FONTSIZE
    plt.rcParams["ytick.labelsize"] = TICK_FONTSIZE

    condition_order = ["Sta", "DevSynC2", "DevSynC1", "DevSem"]
    model_order = [m for m in MODELS if m in set(df["Model"])]
    excluded_models = ["Qwen3-8B-Base"]
    palette = {
        "Sta": "#0072b2",
        "DevSynC2": "#d55e00",
        "DevSynC1": "#009e73",
        "DevSem": "#A65A8A",
    }

    fig = plt.figure(figsize=mm_to_inches(170, 100), constrained_layout=False)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    _plot_panel_A(ax1, df, condition_order, model_order, palette, excluded_models)
    _plot_panel_B(ax2, df, condition_order, palette, excluded_models)

    save_img_path = os.path.join(IMAGE_DIR, "llm_sanity_check.svg")

    plt.savefig(save_img_path, transparent=True, dpi=600, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to {save_img_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLM Sanity Check and Visualization")
    parser.add_argument("--prepare", help="Prepare the dataframe and store it as a .csv file.", action="store_true")
    parser.add_argument("--visualize", help="Generate plots from saved data", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        run_extraction()
    
    if args.visualize:
        run_visualization()
    
    if not args.prepare and not args.visualize:
        parser.print_help()