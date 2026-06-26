import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def extract_llm_features(
    sentence,
    model,
    tokenizer,
    max_history_tokens=None,
    epsilon=1e-16
):
    """
    Calculate the surprisal and precision for each token in a given word squence.

    Parameters
    ----------
    sentence : str
        The input sentence.
    model : CausalLM
        The LLM for forward inference.
    tokenizer : Tokenizer
        A corresponding tokenizer to the LLM.
    max_history_tokens : int (optional)
        The maximum history tokens used to calculate prediction probabilities. Default: None. 
        If None, all the history tokens would be used.
    epsilon : float
        A small number to avoid numerical error when performing logarithmic operations.
        Default : 1e-16
    
    Returns
    -------
    results : list of dict
        Each dictionary corresponds to a token, containing its surprisal and precision.
    
    """
    device = model.device
    
    input_ids = list()
    
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if start_token is not None: input_ids.append(start_token)
    start_calc_idx = 1
    
    sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    
    input_ids.extend(sentence_token_ids)
    input_ids_tensor = torch.tensor([input_ids]).to(device)
    
    results = list()
    for i in range(start_calc_idx, len(input_ids)):
        if max_history_tokens and i > max_history_tokens:
            start_context_index = i - max_history_tokens
            context_ids = input_ids_tensor[:, start_context_index:i]
        else:
            context_ids = input_ids_tensor[:, :i]

        with torch.no_grad():
            outputs = model(context_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]

        probabilities = F.softmax(next_token_logits, dim=-1)

        current_token_id = input_ids[i]
        probability_of_current_token = probabilities[current_token_id].item()
        
        # Surprisal = -logP(x|c)
        surprisal = -np.log(probability_of_current_token + epsilon)
        
        # Entropy = -Σ(P(x|c) * logP(x|c))
        entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon)).item()
        
        # Precision = 1 / Entropy
        precision = 1 / entropy if entropy > 0 else 0.0

        readable_token_text = tokenizer.decode([current_token_id])

        if not readable_token_text.strip() and readable_token_text != " ": 
            readable_token_text = tokenizer.convert_ids_to_tokens(current_token_id)
            if isinstance(readable_token_text, bytes):
                readable_token_text = readable_token_text.decode("utf-8", errors="replace")

        results.append({
            "word": readable_token_text, 
            "surprisal": surprisal, 
            "precision": np.float64(precision)
        })
    
    return results


def extract_candidate_target_probs(
    stem,
    candidate_targets,
    model,
    tokenizer,
    max_history_tokens=None
):
    """
    Calculate the probability of each candidate target conditioned on a stem,
    and compute the restricted probability distribution among candidates.

    Parameters
    ----------
    stem : str
        The context / prefix string.
    candidate_targets : list of str
        Candidate target words or phrases.
    model : CausalLM
        The LLM for forward inference.
    tokenizer : Tokenizer
        A corresponding tokenizer to the LLM.
    max_history_tokens : int, optional
        Maximum number of history tokens used for prediction. If None,
        all history tokens are used.

    Returns
    -------
    results : list of dict
        Each dict contains:
        - target
        - log_probability
        - probability
        - restricted_probability
    """
    device = model.device
    results = list()

    prefix_ids = list()
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if start_token is not None: prefix_ids.append(start_token)

    stem_token_ids = tokenizer.encode(stem, add_special_tokens=False)
    prefix_ids.extend(stem_token_ids)

    def _get_next_token_logprob(context_token_ids, next_token_id):
        """
        Return log P(next_token_id | context_token_ids)
        """
        context_tensor = torch.tensor([context_token_ids], device=device)

        with torch.no_grad():
            outputs = model(context_tensor)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

        return log_probs[next_token_id].item()

    candidate_log_probs = list()
    for target in candidate_targets:
        target_token_ids = tokenizer.encode(target, add_special_tokens=False)

        if len(target_token_ids) == 0:
            raise ValueError(f"Target {repr(target)} is tokenized into an empty sequence.")

        total_log_prob = 0.0
        running_context_ids = prefix_ids.copy()

        for token_id in target_token_ids:
            if max_history_tokens is not None and len(running_context_ids) > max_history_tokens:
                context_ids = running_context_ids[-max_history_tokens:]
            else:
                context_ids = running_context_ids

            token_log_prob = _get_next_token_logprob(context_ids, token_id)
            total_log_prob += token_log_prob

            running_context_ids.append(token_id)

        candidate_log_probs.append(total_log_prob)

        results.append({
            "target": target,
            "log_probability": np.float64(total_log_prob),
            "probability": np.float64(np.exp(total_log_prob))
        })

    log_prob_tensor = torch.tensor(candidate_log_probs, dtype=torch.float64)
    restricted_probs = torch.softmax(log_prob_tensor, dim=0).cpu().numpy()

    for i in range(len(results)):
        results[i]["restricted_probability"] = np.float64(restricted_probs[i])

    return results


def calculate_restricted_entropy(candidate_results, epsilon=1e-16):
    """
    Calculate the entropy of the restricted probability distribution
    over candidate targets.

    Parameters
    ----------
    candidate_results : list of dict
        Output of extract_candidate_target_probs(), where each dict
        contains the key 'restricted_probability'.
    epsilon : float
        Small number for numerical stability.

    Returns
    -------
    entropy : np.float64
        Shannon entropy (base e) of the restricted distribution.
    """
    restricted_probs = np.array(
        [item["restricted_probability"] for item in candidate_results],
        dtype=np.float64
    )

    entropy = -np.sum(restricted_probs * np.log(restricted_probs + epsilon))
    return np.float64(entropy)


if __name__ == "__main__":
    
    # Atom
    model_name = "./huggingface/Atom-7B"

    device_map = "auto"
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map=device_map,
                                                 torch_dtype=torch.float16,
                                                 quantization_config=quantization_config,
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2")
    model = model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    ## Surprisal & Precision
    test = "因为天上在下雨，所以地面很潮湿"
    print(f"\nTest: {test}\n")

    results = extract_llm_features(
        sentence=test,
        model=model,
        tokenizer=tokenizer,
        max_history_tokens=None,
        epsilon=1e-16
    )
    for item in results:
        print(item)

    ## Restricted Prob
    stem = "因为他努力学习，但是成绩很"
    targets = ["优秀", "糟糕"]
    print(f"\nTest: {stem} | {targets[0]}/{targets[1]}\n")

    results = extract_candidate_target_probs(
        stem=stem,
        candidate_targets=targets,
        model=model,
        tokenizer=tokenizer,
        max_history_tokens=None,
    )
    entropy = calculate_restricted_entropy(results)

    print("Candidate results:")
    for item in results:
        print(item)

    print("\nRestricted entropy:", entropy)