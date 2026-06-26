import os
import mne
import numpy as np
import pandas as pd
import argparse
import torch
import gc
import statsmodels.formula.api as smf

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils.configs import TOPIC_SEM_SETTINGS
from utils.llm_features import extract_candidate_target_probs, calculate_restricted_entropy


# Global Parameters
SAVE_ROOT = "./results/7.4-Feature_Extraction"
NUM_TOPICS = 7
BLOCK_SETTINGS = {
    # Based on "TOPIC_SEM_SETTINGS": [context_idx, target_idx, ClauseRelation, C1Type, C2Type]
    "block-1/sent-1" : [0, 0, 1, 1, 1],
    "block-1/sent-2" : [0, 1, -1, 1, 1],
    "block-1/sent-3" : [1, 0, 1, -1, -1],
    "block-1/sent-4" : [1, 1, -1, -1, -1],
    
    "block-2/sent-1" : [0, 0, 1, 1, 1],
    "block-2/sent-2" : [2, 1, -1, 1, -1],
    "block-2/sent-3" : [3, 0, 1, -1, 1],
    "block-2/sent-4" : [1, 1, -1, -1, -1],
    
    "block-3/sent-1" : [0, 0, 1, 1, 1],
    "block-3/sent-2" : [2, 0, 1, 1, -1],
    "block-3/sent-3" : [3, 1, -1, -1, 1],
    "block-3/sent-4" : [1, 1, -1, -1, -1],
}
MODELS = [
    "Atom-7B",
    "Qwen3-0.6B-Base",
    "Qwen3-1.7B-Base",
    "Qwen3-4B-Base",
    "Qwen3-14B-Base"
]
FEATURES = ["ActualSurprisal", "CandidateEntropy", "LocalBias"]
REQ_COLS = ["Model", "SentID", "ActualTarget", "ClauseRelation", 
            "C1Type", "C2Type", "C1_match", "C2_match", "Conflict"] + FEATURES


# Self-defined functions
def run_extraction():
    os.makedirs(SAVE_ROOT, exist_ok=True)

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

            for key in BLOCK_SETTINGS.keys():
                stem_idx = BLOCK_SETTINGS[key][0]
                actual_target_idx = BLOCK_SETTINGS[key][1]

                stem = contexts[stem_idx]
                features = extract_candidate_target_probs(
                    stem=stem,
                    candidate_targets=targets,
                    model=model,
                    tokenizer=tokenizer
                )
                candidate_entropy = calculate_restricted_entropy(features)

                # ClauseRelation (1 for causal, -1 for consessive)
                clause_relation = BLOCK_SETTINGS[key][2]

                # C1Type (1 for "因为", -1 for "虽然")
                c1_type = BLOCK_SETTINGS[key][3]

                # C2Type (1 for "所以", -1 for "但是")
                c2_type = BLOCK_SETTINGS[key][4]

                c1_match = clause_relation * c1_type # C1_match = ClauseRelation × C1Type
                c2_match = clause_relation * c2_type # C2_match = ClauseRelation × C2Type
                conflict = -c1_type * c2_type # Conflict = −(C1Type × C2Type) = −(C1_match × C2_match)

                # LocalBias = logP(local-consistent target∣stem) − logP(global-consistent target∣stem)
                # Only for B2Dev and B3Dev, else NaN
                local_bias = np.nan

                if conflict == 1:
                    # DevSynC2: actual target is locally consistent with C2, but globally inconsistent with C1.
                    if c1_match == -1 and c2_match == 1:
                        local_bias = features[actual_target_idx]["log_probability"] - features[1 - actual_target_idx]["log_probability"]

                    # DevSynC1: actual target is globally consistent with C1, but locally inconsistent with C2.
                    elif c1_match == 1 and c2_match == -1:
                        local_bias = features[1 - actual_target_idx]["log_probability"] - features[actual_target_idx]["log_probability"]

                all_results.append({
                    "Model": model_name,
                    "SentID": key + f"/topic-{topic_idx}",
                    "Stem": stem,
                    "ActualTarget": targets[actual_target_idx],
                    "ActualSurprisal": -features[actual_target_idx]["log_probability"], # Actual Surprisal = −logP(actual target∣stem)
                    "CandidateEntropy": candidate_entropy,
                    "LocalBias": local_bias,
                    "ClauseRelation": clause_relation,
                    "C1Type": c1_type,
                    "C2Type": c2_type,
                    "C1_match": c1_match,
                    "C2_match": c2_match,
                    "Conflict": conflict
                })
            
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    df = pd.DataFrame(all_results)
    save_path = os.path.join(SAVE_ROOT, "stim_llm_features.csv")
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")


def _zscore(s):
    """
    Parameters
    ----------
    s: pd.Series

    Returns
    -------
    pd.Series
    """
    sd = s.std(ddof=0, skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean(skipna=True)) / sd


def _parse_ids(df):
    """
    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # SentID examples: block-2/sent-3/topic-7
    id_parts = df["SentID"].str.extract(r'block-(?P<Block>\d+)/sent-(?P<SentInBlock>\d+)/topic-(?P<TopicID>\d+)')
    for c in id_parts.columns:
        df[c] = pd.to_numeric(id_parts[c], errors="coerce")
    return df


def _infer_condition(row):
    c1 = row["C1_match"]
    c2 = row["C2_match"]
    conflict = row["Conflict"]
    if conflict == -1 and c1 == 1 and c2 == 1:
        return "STA"
    if conflict == -1 and c1 == -1 and c2 == -1:
        return "DevSem"
    if conflict == 1 and c1 == -1 and c2 == 1:
        return "DevSynC2"
    if conflict == 1 and c1 == 1 and c2 == -1:
        return "DevSynC1"
    return "UNKNOWN"


def _add_consensus_features(df):
    """
    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
    """
    # Per-model z-scores so that scales are comparable across LLMs
    for feat in FEATURES:
        df[f"{feat}_z_model"] = df.groupby("Model")[feat].transform(_zscore)
        # Optional topic-centered version to absorb topic base-rate differences
        df[f"{feat}_zc_model_topic"] = (
            df.groupby(["Model", "TopicID"])[feat].transform(lambda x: x - x.mean())
        )
        df[f"{feat}_zczt_model_topic"] = (
            df.groupby("Model")[f"{feat}_zc_model_topic"].transform(_zscore)
        )

    # Consensus = average standardized value across models for the same sentence/item
    gb = df.groupby("SentID", as_index=False)
    consensus = gb.agg(
        ActualTarget=("ActualTarget", "first"),
        ClauseRelation=("ClauseRelation", "first"),
        C1Type=("C1Type", "first"),
        C2Type=("C2Type", "first"),
        C1_match=("C1_match", "first"),
        C2_match=("C2_match", "first"),
        Conflict=("Conflict", "first"),
        Block=("Block", "first"),
        SentInBlock=("SentInBlock", "first"),
        TopicID=("TopicID", "first"),
        Condition=("Condition", "first"),
        n_models=("Model", "nunique"),
    )
    
    for feat in FEATURES:
        mean_df = gb[[f"{feat}_z_model", f"{feat}_zczt_model_topic"]].mean()
        mean_df = mean_df.rename(columns={
            f"{feat}_z_model": f"Consensus_{feat}",
            f"{feat}_zczt_model_topic": f"Consensus_{feat}_topicCentered"
        })
        consensus = consensus.merge(mean_df, on="SentID", how="left")

    consensus["Consensus_LocalBiasAbs"] = consensus["Consensus_LocalBias"].abs()
    return df, consensus


def _fit_models(df):
    """
    Parameters
    ----------
    df: pd.DataFrame

    """
    # Model 1: Main property check -- do LLMs mainly track local consistency (C2_match)
    # and/or global consistency (C1_match), controlling semantic clause relation.
    m1 = smf.mixedlm(
        "ActualSurprisal_z_model ~ C2_match + C1_match + ClauseRelation",
        data=df,
        groups=df["TopicID"],
        vc_formula={"Model": "0 + C(Model)"},
        re_formula="1"
    ).fit(reml=False, method="lbfgs")

    # Model 2: Add interaction to capture conflict / non-conflict structure.
    m2 = smf.mixedlm(
        "ActualSurprisal_z_model ~ C2_match * C1_match + ClauseRelation",
        data=df,
        groups=df["TopicID"],
        vc_formula={"Model": "0 + C(Model)"},
        re_formula="1"
    ).fit(reml=False, method="lbfgs")

    # Optional descriptive model for local-bias magnitude on conflict items only (B2/B3)
    conflict_df = df[df["Conflict"] == 1].copy()
    m3 = smf.mixedlm(
        "LocalBias_z_model ~ C2_match + ClauseRelation",
        data=conflict_df,
        groups=conflict_df["TopicID"],
        vc_formula={"Model": "0 + C(Model)"},
        re_formula="1"
    ).fit(reml=False, method="lbfgs")
    return m1, m2, m3


def run_stats():
    OUT_DIR = os.path.join(SAVE_ROOT, "llm_consensus_outputs")
    os.makedirs(OUT_DIR, exist_ok=True)

    fname = os.path.join(SAVE_ROOT, "stim_llm_features.csv")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing .csv file: {fname}")

    df = pd.read_csv(fname)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.drop_duplicates().copy()
    df = _parse_ids(df)
    df["Condition"] = df.apply(_infer_condition, axis=1)

    # Sanity checks on coding
    if (df["C1_match"].isin([-1, 1]).all() and df["C2_match"].isin([-1, 1]).all() and df["Conflict"].isin([-1, 1]).all()):
        pass
    else:
        raise ValueError("Expected C1_match, C2_match, Conflict to be coded as -1/1.")

    # Build consensus features
    df, consensus = _add_consensus_features(df)
    df.to_csv(os.path.join(OUT_DIR, "stim_llm_features_with_standardized_columns.csv"), index=False)
    consensus.to_csv(os.path.join(OUT_DIR, "stim_llm_consensus_features.csv"), index=False)

    # Fit statsmodels
    m1, m2, m3 = _fit_models(df)

    with open(os.path.join(OUT_DIR, "llm_statsmodel_summaries.txt"), "w", encoding="utf-8") as f:
        f.write("MODEL 1: ActualSurprisal_z_model ~ C2_match + C1_match + ClauseRelation\n")
        f.write(m1.summary().as_text())
        f.write("\n\nMODEL 2: ActualSurprisal_z_model ~ C2_match * C1_match + ClauseRelation\n")
        f.write(m2.summary().as_text())
        f.write("\n\nMODEL 3 (conflict only): LocalBias_z_model ~ C2_match + ClauseRelation\n")
        f.write(m3.summary().as_text())

    # Save fixed-effect tables in .csv form
    for name, model in [("m1", m1), ("m2", m2), ("m3_conflict", m3)]:
        fe_dict = {"estimate": model.params}
        if hasattr(model, "bse"):
            fe_dict["se"] = model.bse
        if hasattr(model, "pvalues"):
            fe_dict["pvalue"] = model.pvalues
            
        fe = pd.DataFrame(fe_dict).reset_index().rename(columns={"index": "term"})
        fe.to_csv(os.path.join(OUT_DIR, f"{name}_fixed_effects.csv"), index=False)

    # Quick descriptive table
    desc = df.groupby(["Model", "Condition"])["ActualSurprisal"].mean().reset_index()
    desc.to_csv(os.path.join(OUT_DIR, "descriptive_actual_surprisal_by_model_condition.csv"), index=False)

    print("Wrote:")
    for p in sorted(os.listdir(OUT_DIR)):
        print(p)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract all needed features from LLMs.")
    parser.add_argument("--prepare", help="Prepare the dataframe and store it as a .csv file.", action="store_true")
    parser.add_argument("--stats", help="Conduct statsmodels and perform the analyses.", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        run_extraction()

    if args.stats:
        run_stats()
    
    if not args.prepare and not args.stats:
        parser.print_help()