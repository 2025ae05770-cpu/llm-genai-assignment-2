import re


JUDGE_PROMPT_TEMPLATE = """You are an impartial examiner. Compare the candidate with the known answer.
2 = correct (all key facts, nothing wrong)
1 = partly correct (some key facts missing, nothing wrong)
0 = wrong
Return exactly:
SCORE: <0, 1, or 2>
REASON: <one sentence>
Query: {query}
Known answer: {known}
Candidate answer: {answer}"""


def parse_judgment(raw):
    score_match = re.search(r"(?im)^\s*SCORE:\s*([012])\s*$", raw)
    reason_match = re.search(r"(?im)^\s*REASON:\s*(.+?)\s*$", raw)
    if not score_match or not reason_match:
        raise ValueError(f"Judge returned an invalid format: {raw}")
    return int(score_match.group(1)), reason_match.group(1).strip()


def judge(llm, query, known, answer):
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, known=known, answer=answer)
    raw, tokens, latency = llm.generate(prompt, temperature=0.0)
    score, reason = parse_judgment(raw)
    return score, reason, tokens, latency, raw
