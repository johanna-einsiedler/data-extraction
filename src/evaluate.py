import re


def clean_answer(ans: str) -> str:
    """Extract first meaningful token/letter from LLM output."""
    if not ans:
        return ""
    ans = ans.strip()
    ans = re.sub(r"^[^\w]*", "", ans)  # remove leading non-word chars
    ans = re.sub(r"[^\w]*$", "", ans)  # remove trailing non-word chars
    first_token = ans.split()[0]
    return first_token.upper()  # normalize


def evaluate_answer(llm_output: dict, true_answer: str):
    """
    Evaluate a single LLM output against the true answer.

    Parameters:
    - llm_output: dict containing 'text' (str) and optionally 'logprobs' (list/float)
    - true_answer: correct answer (str)

    Returns:
    - dict with keys: cleaned_answer, true_answer, accuracy, logprobs
    """
    raw_answer = llm_output.get("text", "")
    logprobs = llm_output.get("logprobs", None)
    cleaned = clean_answer(raw_answer)
    true_upper = true_answer.upper() if true_answer else ""
    acc = int(cleaned == true_upper)

    return {
        "raw_answer": raw_answer,
        "cleaned_answer": cleaned,
        "true_answer": true_upper,
        "accuracy": acc,
        "logprobs": logprobs,
    }
