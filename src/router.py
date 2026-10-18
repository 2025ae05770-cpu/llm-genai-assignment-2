from .patterns import chain_of_thought, plan_solve_verify, structured, zero_shot
from .react import react


ROUTE_LABELS = (
    "SIMPLE_FACTUAL",
    "STRUCTURED_EXTRACTION",
    "MULTI_STEP_REASONING",
    "NEEDS_TOOL",
    "HIGH_STAKES",
)

CLASSIFIER_PROMPT_V1 = """Classify the query as exactly one label below. Return only the label.
SIMPLE_FACTUAL: a direct factual question needing no special format or multi-step work.
STRUCTURED_EXTRACTION: asks to extract specific fields or return structured data.
MULTI_STEP_REASONING: requires multiple reasoning steps but no external tool.
NEEDS_TOOL: requires calculation or a local tool.
HIGH_STAKES: a consequential answer that should be checked before responding.
Query: {query}"""

CLASSIFIER_PROMPT_V2 = """Choose the single best route for the query. Return exactly one label,
with no punctuation, explanation, or other text:
SIMPLE_FACTUAL = a simple fact or direct answer.
STRUCTURED_EXTRACTION = extract named fields or produce structured output.
MULTI_STEP_REASONING = combine facts through multiple steps; no calculation tool is needed.
NEEDS_TOOL = a numeric calculation or tool action is required.
HIGH_STAKES = a consequential answer where an explicit verification pass is needed.
If types overlap, prefer HIGH_STAKES, then NEEDS_TOOL, then STRUCTURED_EXTRACTION,
then MULTI_STEP_REASONING, then SIMPLE_FACTUAL.
Query: {query}"""


def classify(llm, query, version=1):
    if version not in (1, 2):
        raise ValueError("Classifier prompt version must be 1 or 2.")
    template = CLASSIFIER_PROMPT_V1 if version == 1 else CLASSIFIER_PROMPT_V2
    raw, tokens, latency = llm.generate(template.format(query=query), temperature=0.0)
    label = raw.strip()
    if label not in ROUTE_LABELS:
        raise ValueError(f"Classifier returned an unknown label: {raw!r}")
    return label, tokens, latency, template.format(query=query)


def route(llm, query, classifier_version=1):
    label, classifier_tokens, classifier_latency, _ = classify(
        llm, query, version=classifier_version
    )
    if label == "STRUCTURED_EXTRACTION":
        answer, tokens, latency = structured(llm, query)
    elif label == "MULTI_STEP_REASONING":
        answer, tokens, latency = chain_of_thought(llm, query)
    elif label == "NEEDS_TOOL":
        answer, tokens, latency = react(llm, query)
    elif label == "HIGH_STAKES":
        answer, tokens, latency = plan_solve_verify(llm, query)
    else:
        answer, tokens, latency = zero_shot(llm, query)
    return answer, label, tokens + classifier_tokens, latency + classifier_latency
