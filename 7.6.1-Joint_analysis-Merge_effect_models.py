import os
import warnings
import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------
# Helpers
# -----------------------------
def zscore_series(s: pd.Series):
    sd = s.std(ddof=0, skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean(skipna=True)) / sd


def parse_sentid(df: pd.DataFrame):
    # SentID format: block-2/sent-3/topic-7
    parts = df["SentID"].str.extract(
        r"block-(?P<Block>\d+)/sent-(?P<SentInBlock>\d+)/topic-(?P<TopicID>\d+)"
    )
    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")
    for c in parts.columns:
        if c not in df.columns:
            df[c] = parts[c]
        else:
            df[c] = df[c].fillna(parts[c])
    return df


def infer_condition(row):
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


def fit_mixedlm(formula, data, group_col="sid", vc_item=True, model_name="model"):
    """
    Try fitting mixedlm with:
      random intercept for subject (groups=group_col)
      + optional variance component for item(SentID)
    """
    try:
        if vc_item:
            m = smf.mixedlm(
                formula=formula,
                data=data,
                groups=data[group_col],
                vc_formula={"SentID": "0 + C(SentID)"},
                re_formula="1"
            ).fit(reml=False, method="lbfgs")
        else:
            m = smf.mixedlm(
                formula=formula,
                data=data,
                groups=data[group_col],
                re_formula="1"
            ).fit(reml=False, method="lbfgs")
        return m, None
    except Exception as e1:
        # fallback: no item vc
        try:
            m = smf.mixedlm(
                formula=formula,
                data=data,
                groups=data[group_col],
                re_formula="1"
            ).fit(reml=False, method="lbfgs")
            return m, f"[WARN] {model_name}: fallback without item VC due to: {e1}"
        except Exception as e2:
            return None, f"[ERROR] {model_name}: failed. primary={e1}; fallback={e2}"


def save_mixedlm_outputs(model, out_dir, name):
    if model is None:
        return
    # summary txt
    with open(os.path.join(out_dir, f"{name}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    # fixed effects csv
    fe = pd.DataFrame({
        "term": model.fe_params.index,
        "estimate": model.fe_params.values,
        "se": model.bse_fe.reindex(model.fe_params.index).values,
        "zvalue": (model.fe_params / model.bse_fe.reindex(model.fe_params.index)).values,
        "pvalue": model.pvalues.reindex(model.fe_params.index).values,
    })
    fe.to_csv(os.path.join(out_dir, f"{name}_fixed_effects.csv"), index=False)


def save_ols_outputs(model, out_dir, name):
    if model is None:
        return
    # summary txt
    with open(os.path.join(out_dir, f"{name}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    
    # coefficients csv
    coefs = pd.DataFrame({
        "term": model.params.index,
        "estimate": model.params.values,
        "se": model.bse.values,
        "tvalue": model.tvalues.values,
        "pvalue": model.pvalues.values,
    })
    coefs.to_csv(os.path.join(out_dir, f"{name}_coefficients.csv"), index=False)


def term_table(model, model_name, formula, model_type="MixedLM"):
    if model is None:
        return pd.DataFrame()
    
    idx = model.params.index
    out = pd.DataFrame({
        "model": model_name,
        "type": model_type,
        "formula": formula,
        "term": idx,
        "estimate": model.params.values,
        "se": getattr(model, "bse", pd.Series(np.nan, index=idx)).loc[idx].values,
        "pvalue": getattr(model, "pvalues", pd.Series(np.nan, index=idx)).loc[idx].values,
    })
    return out


def model_fit_row(model, model_name, formula, model_type="MixedLM"):
    vals = {
        "model": model_name,
        "type": model_type,
        "formula": formula,
        "nobs": getattr(model, "nobs", np.nan),
        "aic": getattr(model, "aic", np.nan),
        "bic": getattr(model, "bic", np.nan),
        "llf": getattr(model, "llf", np.nan),
    }
    if hasattr(model, "rsquared"):
        vals["r2"] = model.rsquared
    if hasattr(model, "rsquared_adj"):
        vals["r2_adj"] = model.rsquared_adj
    return vals


# -----------------------------
# Main
# -----------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if_zscored = not args.no_zscore

    models_info = []
    fixed_tables = []
    fit_rows = []

    llm = pd.read_csv(args.llm_csv)
    nb = pd.read_csv(args.neuro_csv)
    nb = nb.loc[:, ~nb.columns.str.contains("^Unnamed")]

    llm = parse_sentid(llm.drop_duplicates().copy())
    nb = parse_sentid(nb.drop_duplicates().copy())

    if "Condition" not in llm.columns:
        llm["Condition"] = llm.apply(infer_condition, axis=1)

    df = nb.merge(llm, on="SentID", how="inner", suffixes=("", "_llm"))
    for c in ["ClauseRelation", "C1Type", "C2Type", "C1_match", "C2_match", "Conflict", "Condition", "TopicID", "Block", "SentInBlock"]:
        c_llm = f"{c}_llm"
        if c_llm in df.columns:
            df[c] = df[c_llm]
            df.drop(columns=[c_llm], inplace=True)

    base_map = {
        "ActualSurprisal": "Consensus_ActualSurprisal",
        "CandidateEntropy": "Consensus_CandidateEntropy",
        "LocalBias": "Consensus_LocalBias"
    }

    feat_cols = {}
    for k, v in base_map.items():
        col = v + "_topicCentered" if args.use_topic_centered else v
        if col not in df.columns:
            raise ValueError(f"Missing LLM feature column: {col}")
        feat_cols[k] = col

    df["LLM_ActualSurprisal"] = df[feat_cols["ActualSurprisal"]]
    df["LLM_CandidateEntropy"] = df[feat_cols["CandidateEntropy"]]
    df["LLM_LocalBias"] = df[feat_cols["LocalBias"]]
    df["LLM_LocalBiasAbs"] = df["LLM_LocalBias"].abs()

    keep_cols = [
        "sid", "SentID", "RT", "Run",
        "Pre_MTG_LH_amp", "Post_400_MidTL_LH_amp", "Post_400_ATL_LH_amp",
        "ClauseRelation", "C1Type", "C2Type", "C1_match", "C2_match", "Conflict",
        "Condition", "TopicID", "Block", "SentInBlock",
        "LLM_ActualSurprisal", "LLM_CandidateEntropy", "LLM_LocalBias", "LLM_LocalBiasAbs"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = df.dropna(subset=["sid", "SentID", "RT", 
                           "Pre_MTG_LH_amp", "Post_400_MidTL_LH_amp", "Post_400_ATL_LH_amp",
                           "LLM_ActualSurprisal", "LLM_CandidateEntropy"]).copy()

    rt_mean = df["RT"].mean()
    rt_std = df["RT"].std()
    cutoff_high = rt_mean + 3 * rt_std
    cutoff_low = 200

    original_count = len(df)
    df = df[(df["RT"] >= cutoff_low) & (df["RT"] <= cutoff_high)].copy()
    cleaned_count = len(df)
    print(f"Removed {original_count - cleaned_count} trials ({((original_count - cleaned_count)/original_count)*100:.2f}%)")

    if args.use_log_rt:
        df = df[df["RT"] > 0].copy()
        df["RT_used"] = np.log(df["RT"])
    else:
        df["RT_used"] = df["RT"]

    for c in ["ClauseRelation", "C1Type", "C2Type", "C1_match", "C2_match", "Conflict"]:
        if c in df.columns:
            bad = ~df[c].isin([-1, 1])
            if bad.any():
                raise ValueError(f"{c} contains non -1/1 values, please check coding.")

    # DevSyn Conflict
    df["Cond_DevSyn"] = np.nan
    df.loc[df["Condition"] == "DevSynC2", "Cond_DevSyn"] = -1
    df.loc[df["Condition"] == "DevSynC1", "Cond_DevSyn"] = 1

    cont_cols = [
        "Pre_MTG_LH_amp", "Post_400_MidTL_LH_amp", "Post_400_ATL_LH_amp",
        "RT_used", "Run",
        "LLM_ActualSurprisal", "LLM_CandidateEntropy", "LLM_LocalBias", "LLM_LocalBiasAbs"
    ]
    if if_zscored:
        for c in cont_cols:
            if c == "Run": continue
            if c in df.columns:
                df[f"{c}_z"] = zscore_series(df[c].astype(float))
        # Use z versions in formulas
        pre_mtg = "Pre_MTG_LH_amp_z"
        post_MidTL = "Post_400_MidTL_LH_amp_z"
        post_atl = "Post_400_ATL_LH_amp_z"
        rt_y = "RT_used_z"
        run_x = "Run_z" if "Run_z" in df.columns else "Run"
        llm_surp = "LLM_ActualSurprisal_z"
        llm_ent = "LLM_CandidateEntropy_z"
        llm_lb = "LLM_LocalBias_z"
    else:
        pre_mtg = "Pre_MTG_LH_amp"
        post_MidTL = "Post_400_MidTL_LH_amp"
        post_atl = "Post_400_ATL_LH_amp"
        rt_y = "RT_used"
        run_x = "Run"
        llm_surp = "LLM_ActualSurprisal"
        llm_ent = "LLM_CandidateEntropy"
        llm_lb = "LLM_LocalBias"

    logs = []

    # =========================================================
    # Entropy-as-structure-induced-uncertainty analyses
    # =========================================================
    # Item-level table: LLM features are stimulus-level, so structure -> entropy is fitted on unique SentID.
    item_df = df.drop_duplicates("SentID").copy()
    item_df["LLM_CandidateEntropy_z"] = zscore_series(item_df["LLM_CandidateEntropy"].astype(float))
    item_df["LLM_ActualSurprisal_z"] = zscore_series(item_df["LLM_ActualSurprisal"].astype(float))

    ## A1: Does connective configuration induce candidate uncertainty?
    ## Model: LLM certainty ~ C1Type * C2Type + ClauseRelation
    formula = "LLM_CandidateEntropy_z ~ C1Type * C2Type + ClauseRelation"
    m_ent_struct = smf.ols(formula, data=item_df).fit(cov_type="HC3")
    save_ols_outputs(m_ent_struct, args.out_dir, "entropy_structure_ols")
    models_info.append(("entropy_structure_ols", formula, m_ent_struct))
    fixed_tables.append(term_table(m_ent_struct, "entropy_structure_ols", formula, model_type="OLS_HC3"))
    fit_rows.append(model_fit_row(m_ent_struct, "entropy_structure_ols", formula, model_type="OLS_HC3"))

    ## Descriptive condition means for candidate entropy.
    ent_condition_summary = item_df.groupby("Condition").agg(
        n=("SentID", "count"),
        mean_entropy=("LLM_CandidateEntropy", "mean"),
        sd_entropy=("LLM_CandidateEntropy", "std"),
        mean_entropy_z=("LLM_CandidateEntropy_z", "mean"),
        sd_entropy_z=("LLM_CandidateEntropy_z", "std"),
    ).reset_index()
    ent_condition_summary.to_csv(os.path.join(args.out_dir, "entropy_by_condition.csv"), index=False)

    ## Split entropy into structure-predicted and residual components, then map to trial data.
    item_df["Entropy_struct_component"] = m_ent_struct.fittedvalues
    item_df["Entropy_residual_component"] = m_ent_struct.resid
    item_df["Entropy_struct_component_z"] = zscore_series(item_df["Entropy_struct_component"])
    item_df["Entropy_residual_component_z"] = zscore_series(item_df["Entropy_residual_component"])
    comp_cols = ["SentID", "Entropy_struct_component", "Entropy_residual_component",
                 "Entropy_struct_component_z", "Entropy_residual_component_z"]
    df = df.merge(item_df[comp_cols], on="SentID", how="left")
    item_df.to_csv(os.path.join(args.out_dir, "item_level_entropy_components.csv"), index=False)

    # Save merged DataFrame
    df.to_csv(os.path.join(args.out_dir, "merged_trial_llm_neuro_behav.csv"), index=False)

    # -----------------------------
    # PRE-TARGET MODELS
    # -----------------------------
    ## B1. Descriptive structure-only model
    ## Model: pre ~ C1Type * C2Type + Run
    formula = f"{pre_mtg} ~ C1Type * C2Type + {run_x}"
    m_pre_struct, msg = fit_mixedlm(
        formula=formula,
        data=df, group_col="sid", vc_item=True, model_name="pre_MTG_structure_only"
    )
    if msg: logs.append(msg)
    save_mixedlm_outputs(m_pre_struct, args.out_dir, "pre_MTG_structure_only")
    if m_pre_struct is not None:
        models_info.append(("pre_MTG_structure_only", formula, m_pre_struct))
        fixed_tables.append(term_table(m_pre_struct, "pre_MTG_structure_only", formula))
        fit_rows.append(model_fit_row(m_pre_struct, "pre_MTG_structure_only", formula))

    ## B2. Entropy-only neural association model
    ## Model: pre ~ LLM uncertainty + Run
    formula = f"{pre_mtg} ~ {llm_ent} + {run_x}"
    m_pre_ent, msg = fit_mixedlm(
        formula=formula,
        data=df, group_col="sid", vc_item=True, model_name="pre_MTG_entropy_only"
    )
    if msg: logs.append(msg)
    save_mixedlm_outputs(m_pre_ent, args.out_dir, "pre_MTG_entropy_only")
    if m_pre_ent is not None:
        models_info.append(("pre_MTG_entropy_only", formula, m_pre_ent))
        fixed_tables.append(term_table(m_pre_ent, "pre_MTG_entropy_only", formula))
        fit_rows.append(model_fit_row(m_pre_ent, "pre_MTG_entropy_only", formula))

    ## B3. Shared-variance model: structure and entropy compete for variance.
    ## Model: pre ~ C1Type * C2Type + LLM uncertainty + Run
    formula = f"{pre_mtg} ~ C1Type * C2Type + {llm_ent} + {run_x}"
    m_pre_shared, msg = fit_mixedlm(
        formula=formula,
        data=df, group_col="sid", vc_item=True, model_name="pre_MTG_structure_plus_entropy"
    )
    if msg: logs.append(msg)
    save_mixedlm_outputs(m_pre_shared, args.out_dir, "pre_MTG_structure_plus_entropy")
    if m_pre_shared is not None:
        models_info.append(("pre_MTG_structure_plus_entropy", formula, m_pre_shared))
        fixed_tables.append(term_table(m_pre_shared, "pre_MTG_structure_plus_entropy", formula))
        fit_rows.append(model_fit_row(m_pre_shared, "pre_MTG_structure_plus_entropy", formula))

    ## B4. Structural vs residual entropy components
    if if_zscored:
        ent_struct = "Entropy_struct_component_z"
        ent_resid = "Entropy_residual_component_z"
    else:
        ent_struct = "Entropy_struct_component"
        ent_resid = "Entropy_residual_component"
    formula = f"{pre_mtg} ~ {ent_struct} + {ent_resid} + {run_x}"
    m_pre_comp, msg = fit_mixedlm(
        formula=formula,
        data=df, group_col="sid", vc_item=True, model_name="pre_MTG_entropy_structural_vs_residual"
    )
    if msg: logs.append(msg)
    save_mixedlm_outputs(m_pre_comp, args.out_dir, "pre_MTG_entropy_structural_vs_residual")
    if m_pre_comp is not None:
        models_info.append(("pre_MTG_entropy_structural_vs_residual", formula, m_pre_comp))
        fixed_tables.append(term_table(m_pre_comp, "pre_MTG_entropy_structural_vs_residual", formula))
        fit_rows.append(model_fit_row(m_pre_comp, "pre_MTG_entropy_structural_vs_residual", formula))

    # -----------------------------
    # POST-TARGET MODELS
    # -----------------------------
    # Model: post ~ pre * surprisal + C1_match * C2_match + ClauseRelation + Run
    formula = f"{post_MidTL} ~ {pre_mtg} * {llm_surp} + C1_match * C2_match + ClauseRelation + {run_x}"
    m_post1, msg = fit_mixedlm(
        formula=formula,
        data=df, group_col="sid", vc_item=True, model_name="post_MidTL_main"
    )
    if msg: logs.append(msg)
    save_mixedlm_outputs(m_post1, args.out_dir, "post_MidTL_main")
    if m_post1 is not None:
        models_info.append(("post_MidTL_main", formula, m_post1))
        fixed_tables.append(term_table(m_post1, "post_MidTL_main", formula))
        fit_rows.append(model_fit_row(m_post1, "post_MidTL_main", formula))

    formula = f"{post_atl} ~ {pre_mtg} * {llm_surp} + C1_match * C2_match + ClauseRelation + {run_x}"
    m_post2, msg = fit_mixedlm(
        formula=formula,
        data=df, group_col="sid", vc_item=True, model_name="post_ATL_main"
    )
    if msg: logs.append(msg)
    save_mixedlm_outputs(m_post2, args.out_dir, "post_ATL_main")
    if m_post2 is not None:
        models_info.append(("post_ATL_main", formula, m_post2))
        fixed_tables.append(term_table(m_post2, "post_ATL_main", formula))
        fit_rows.append(model_fit_row(m_post2, "post_ATL_main", formula))

    # -----------------------------
    # DevSynC2/DevSynC1 SUBSET MODELS
    # -----------------------------
    dev_syn = df[df["Condition"].isin(["DevSynC2", "DevSynC1"])].copy()
    dev_syn = dev_syn.dropna(subset=[
        "Cond_DevSyn", llm_lb,
        pre_mtg, post_MidTL, rt_y,
        "ClauseRelation", run_x
    ]).copy()
    if len(dev_syn) > 10:

        # post in conflict items
        formula = f"{post_MidTL} ~ {pre_mtg} + {llm_lb} * Cond_DevSyn + ClauseRelation + {run_x}"
        m_dev_syn_post, msg = fit_mixedlm(
            formula=formula,
            data=dev_syn, group_col="sid", vc_item=True, model_name="dev_syn_post_MidTL"
        )
        if msg: logs.append(msg)
        save_mixedlm_outputs(m_dev_syn_post, args.out_dir, "dev_syn_post_MidTL")
        if m_dev_syn_post is not None:
            models_info.append(("dev_syn_post_MidTL", formula, m_dev_syn_post))
            fixed_tables.append(term_table(m_dev_syn_post, "dev_syn_post_MidTL", formula))
            fit_rows.append(model_fit_row(m_dev_syn_post, "dev_syn_post_MidTL", formula))

        # rt in conflict items
        formula = f"{rt_y} ~ {pre_mtg} + {post_MidTL} + {llm_lb} * Cond_DevSyn + ClauseRelation + {run_x}"
        m_dev_syn_rt, msg = fit_mixedlm(
            formula=formula,
            data=dev_syn, group_col="sid", vc_item=True, model_name="dev_syn_rt"
        )
        if msg: logs.append(msg)
        save_mixedlm_outputs(m_dev_syn_rt, args.out_dir, "dev_syn_rt")
        if m_dev_syn_rt is not None:
            models_info.append(("dev_syn_rt", formula, m_dev_syn_rt))
            fixed_tables.append(term_table(m_dev_syn_rt, "dev_syn_rt", formula))
            fit_rows.append(model_fit_row(m_dev_syn_rt, "dev_syn_rt", formula))
    else:
        logs.append("[WARN] DevSynC2/DevSynC1 subset too small, skipped DevSyn models.")

    # -----------------------------
    # Save run config, model summaries, and logs
    # -----------------------------
    cfg = {
        "llm_csv": args.llm_csv,
        "neuro_csv": args.neuro_csv,
        "out_dir": args.out_dir,
        "use_topic_centered": args.use_topic_centered,
        "zscore": if_zscored,
        "use_log_rt": args.use_log_rt
    }
    pd.Series(cfg).to_csv(os.path.join(args.out_dir, "run_config.csv"), header=["value"])

    if fixed_tables:
        pd.concat(fixed_tables, ignore_index=True).to_csv(os.path.join(args.out_dir, "all_fixed_effects_compact.csv"), index=False)
    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(os.path.join(args.out_dir, "all_model_fit_indices.csv"), index=False)
    
    if models_info:
        with open(os.path.join(args.out_dir, "all_models_summary.txt"), "w", encoding="utf-8") as f:
            for idx, (name, formula, model) in enumerate(models_info, start=1):
                f.write(f"MODEL {idx}: {name}\n")
                f.write(f"Formula: {formula}\n")
                f.write(model.summary().as_text())
                f.write("\n\n" + "-" * 80 + "\n\n")
        print(f"All models summary saved to {os.path.join(args.out_dir, 'all_models_summary.txt')}")
    else:
        logs.append("[WARN] No models were successfully fitted.")

    with open(os.path.join(args.out_dir, "fit_logs.txt"), "w", encoding="utf-8") as f:
        if len(logs) == 0:
            f.write("No warnings.\n")
        else:
            for line in logs:
                f.write(str(line) + "\n")

    print("Done.")
    print(f"Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Merge consensus LLM features with trial-level neuro/behavior data and fit mixed models."
    )
    parser.add_argument("--llm_csv", type=str, 
                        default="./results/7.4-Feature_Extraction/llm_consensus_outputs/stim_llm_consensus_features.csv",
                        help="Path to stim_llm_consensus_features.csv")
    parser.add_argument("--neuro_csv", type=str, 
                        default="./results/7.5-Neuro_Behavior_Extraction/ROISourceAmpRT_PrePostTarget.csv",
                        help="Path to ROISourceAmpRT_PrePostTarget.csv")
    parser.add_argument("--out_dir", type=str, default="./results/7.6-Effect_models/meg_llm_model_outputs",
                        help="Output directory")
    parser.add_argument("--use_topic_centered", action="store_true",
                        help="Use *_topicCentered consensus features. Not recommended.")
    parser.add_argument("--no_zscore", action="store_true",
                        help="Disable Z-score normalization (enabled by default)")
    parser.add_argument("--use_log_rt", action="store_true",
                        help="Use log(RT) as dependent variable in RT models")

    args = parser.parse_args()
    main(args)