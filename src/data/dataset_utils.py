import numpy as np
from datasets import load_dataset

def preprocess_dataset(wikitext, nq, tokenizer, num_samples=20):
    """Preprocess WikiText and Natural Questions datasets for evaluation."""
    # WikiText: Filter sequences between 20 and 50 words
    wikitext_sequences = [entry["text"] for entry in wikitext if 20 <= len(entry["text"].split()) <= 50]
    wikitext_samples = np.random.choice(
        wikitext_sequences, min(num_samples // 2, len(wikitext_sequences)), replace=False
    ).tolist()
    
    # Natural Questions: Extract question-answer pairs between 20 and 50 words
    nq_samples = []
    for entry in nq:
        question = entry["question"]["text"]
        short_answers = entry["annotations"]["short_answers"]
        if short_answers and short_answers[0]["text"]:
            answer_raw = short_answers[0]["text"]
            if isinstance(answer_raw, list):
                if not answer_raw or not isinstance(answer_raw[0], str):
                    continue
                answer = answer_raw[0]
            elif isinstance(answer_raw, str):
                answer = answer_raw
            else:
                continue
            total_len = len(question.split()) + len(answer.split())
            if 20 <= total_len <= 50:
                input_ids = tokenizer(question, return_tensors="pt", truncation=True, max_length=50).input_ids[0]
                nq_samples.append({"input": question, "answer": answer, "input_len": len(input_ids)})
        if len(nq_samples) >= num_samples // 2:
            break
    
    # Combine and shuffle samples
    dataset_subset = [
        {"input": s, "answer": None, "input_len": len(tokenizer(s, return_tensors="pt", truncation=True, max_length=50).input_ids[0])}
        for s in wikitext_samples
    ] + nq_samples[:num_samples // 2]
    np.random.shuffle(dataset_subset)
    
    # Cache ground truth token IDs
    ground_truth_cache = []
    for sample in dataset_subset:
        if sample["answer"]:
            answer_ids = tokenizer(sample["answer"], return_tensors="pt", truncation=True, max_length=50).input_ids[0]
            ground_truth_cache.append(answer_ids.tolist())
        else:
            ground_truth_cache.append(None)
    
    return dataset_subset, ground_truth_cache

def load_and_preprocess_data(tokenizer, num_samples=20):
    """Load datasets and preprocess them for model evaluation."""
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    nq = load_dataset("natural_questions", split="train")
    dataset_subset, ground_truth_cache = preprocess_dataset(wikitext, nq, tokenizer, num_samples)
    return dataset_subset, ground_truth_cache