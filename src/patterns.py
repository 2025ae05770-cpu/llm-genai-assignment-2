import json
import re

from config import STRUCTURED_FIELDS

DEFAULT_STRUCTURED_FIELDS = STRUCTURED_FIELDS


def zero_shot(llm, query):
    return llm.generate(query)


def few_shot(llm, query, examples):
    if len(examples) != 3:
        raise ValueError("Few-shot prompting requires exactly three examples.")
    prompt = "\n\n".join(
        f"Question: {example['query']}\nAnswer: {example['answer']}"
        for example in examples
    )
    return llm.generate(f"{prompt}\n\nQuestion: {query}\nAnswer:")


def instruction(llm, query):
    prompt = (
        "You are a precise domain assistant. Follow these steps:\n"
        "1. Identify exactly what the question asks.\n"
        "2. Use only relevant facts and state uncertainty when needed.\n"
        "3. Give a concise, direct answer.\n"
        f"Question: {query}\nAnswer:"
    )
    return llm.generate(prompt)


def self_critique(llm, query, return_details=False):
    first_answer, tokens1, latency1 = llm.generate(query)
    critique, tokens2, latency2 = llm.generate(
        f"Question: {query}\nDraft answer: {first_answer}\n"
        "Check the draft for factual errors, unsupported claims, and missing key facts. "
        "Give a concise critique."
    )
    revised, tokens3, latency3 = llm.generate(
        f"Question: {query}\nDraft answer: {first_answer}\nCritique: {critique}\n"
        "Revise the draft using the critique. Return only the final answer."
    )
    result = (revised, tokens1 + tokens2 + tokens3, latency1 + latency2 + latency3)
    if return_details:
        return (*result, (first_answer, critique, revised))
    return result


def parse_structured(raw, required_fields=DEFAULT_STRUCTURED_FIELDS):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    missing = [field for field in required_fields if field not in value]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    return value


def structured(llm, query, retry=True, required_fields=DEFAULT_STRUCTURED_FIELDS,
               return_details=False):
    fields = ", ".join(required_fields)
    prompt = (
        f"Return one JSON object only, with these required keys: {fields}.\n"
        f"Question: {query}"
    )
    raw, tokens, latency = llm.generate(prompt)
    error = None
    try:
        parsed = parse_structured(raw, required_fields)
    except (json.JSONDecodeError, ValueError) as exc:
        error = str(exc)
        if not retry:
            raise ValueError(f"Structured output was invalid: {error}") from exc
        retry_prompt = (
            f"Your previous response was invalid JSON or did not match the required fields.\n"
            f"Validation error: {error}\n"
            f"Required keys: {fields}\n"
            "Return only a corrected JSON object.\n"
            f"Question: {query}\nPrevious response: {raw}"
        )
        raw, retry_tokens, retry_latency = llm.generate(retry_prompt)
        tokens += retry_tokens
        latency += retry_latency
        parsed = parse_structured(raw, required_fields)
    answer = json.dumps(parsed, ensure_ascii=False)
    result = (answer, tokens, latency)
    if return_details:
        return (*result, {"parsed": parsed, "raw": raw, "initial_error": error})
    return result


def chain_of_thought(llm, query, temperature=0.0):
    prompt = (
        "Solve the problem carefully in multiple steps. Explain the reasoning, "
        "then end with exactly one line in the form 'Final answer: ...'.\n"
        f"Problem: {query}"
    )
    return llm.generate(prompt, temperature)


def final_answer(text):
    matches = re.findall(r"Final answer:\s*(.+)", text, re.IGNORECASE)
    return matches[-1].strip() if matches else text.strip()


def self_consistency(llm, query, samples=5, return_details=False):
    if samples != 5:
        raise ValueError("The assignment requires exactly five self-consistency samples.")
    answers = []
    tokens = 0
    latency = 0.0
    for _ in range(samples):
        text, used_tokens, seconds = chain_of_thought(llm, query, temperature=0.7)
        answers.append(final_answer(text))
        tokens += used_tokens
        latency += seconds
    counts = {answer: answers.count(answer) for answer in answers}
    highest_count = max(counts.values())
    winner = next(answer for answer in answers if counts[answer] == highest_count)
    result = (winner, tokens, latency)
    if return_details:
        return (*result, answers)
    return result


def plan_solve_verify(llm, query, return_details=False):
    plan, tokens1, latency1 = llm.generate(
        f"Write a concise solution plan only. Do not solve yet.\nProblem: {query}"
    )
    solution, tokens2, latency2 = llm.generate(
        f"Problem: {query}\nPlan: {plan}\nFollow the plan and solve the problem."
    )
    critique, tokens3, latency3 = llm.generate(
        f"Problem: {query}\nDraft solution: {solution}\n"
        "Check it for errors, unsupported assumptions, and omissions. Give a concise critique."
    )
    verified, tokens4, latency4 = llm.generate(
        f"Problem: {query}\nDraft solution: {solution}\nCritique: {critique}\n"
        "Revise as needed and return only the verified final answer."
    )
    result = (
        verified,
        tokens1 + tokens2 + tokens3 + tokens4,
        latency1 + latency2 + latency3 + latency4,
    )
    if return_details:
        return (*result, (plan, solution, critique, verified))
    return result
