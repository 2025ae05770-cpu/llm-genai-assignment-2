import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    ADAPTER_PATH,
    BENCHMARK_SIZE,
    CLASSIFIER_CASES,
    DATASET_PATH,
    FEW_SHOT_COUNT,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    REACT_QUERY,
    REASONING_PROBLEMS,
    SEED,
    STRUCTURED_FIELDS,
)
from src.judge import JUDGE_PROMPT_TEMPLATE, judge
from src.patterns import (
    chain_of_thought,
    final_answer,
    few_shot,
    instruction,
    parse_structured,
    plan_solve_verify,
    self_consistency,
    self_critique,
    structured,
    zero_shot,
)
from src.react import react
from src.router import CLASSIFIER_PROMPT_V1, CLASSIFIER_PROMPT_V2, classify, route


PATTERN_NAMES = ("Zero-shot", "Few-shot", "Instruction", "Self-critique", "Structured")
ROUTE_LABELS = (
    "SIMPLE_FACTUAL",
    "STRUCTURED_EXTRACTION",
    "MULTI_STEP_REASONING",
    "NEEDS_TOOL",
    "HIGH_STAKES",
)


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. The Assignment 1 dataset should be at "
            "'Assignment-1/instruction_dataset.jsonl'."
        )
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Dataset row {line_number} must be a JSON object.")
            query = record.get("query", record.get("instruction"))
            answer = record.get("answer", record.get("response"))
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Dataset row {line_number} has no non-empty query/instruction.")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Dataset row {line_number} has no non-empty answer/response.")
            records.append(
                {
                    **record,
                    "query": query.strip(),
                    "answer": answer.strip(),
                }
            )
    minimum = BENCHMARK_SIZE + FEW_SHOT_COUNT
    if len(records) < minimum:
        raise ValueError(
            f"Need at least {minimum} valid examples for {BENCHMARK_SIZE} benchmark "
            f"queries and {FEW_SHOT_COUNT} disjoint few-shot examples; found {len(records)}."
        )
    return records


def validate_setup():
    records = load_jsonl(DATASET_PATH)
    if not GENERATOR_MODEL or not JUDGE_MODEL:
        raise ValueError("Set GENERATOR_MODEL and JUDGE_MODEL in config.py.")
    if GENERATOR_MODEL == JUDGE_MODEL:
        raise ValueError("The assignment requires a judge model different from the generator.")
    if ADAPTER_PATH and not Path(ADAPTER_PATH).is_dir():
        raise FileNotFoundError(
            f"Assignment 1 adapter directory is missing: {ADAPTER_PATH}. "
            "Copy qlora_adapter_B from the Assignment 1 training environment, or set "
            "ADAPTER_PATH = None to explicitly run the unadapted 4-bit base model."
        )
    if len(REASONING_PROBLEMS) != 2 or any(
        not item.get("query") or not item.get("answer") for item in REASONING_PROBLEMS
    ):
        raise ValueError("Configure exactly two reasoning problems with known answers.")
    if not REACT_QUERY:
        raise ValueError("Configure REACT_QUERY in config.py.")
    if len(CLASSIFIER_CASES) != len(ROUTE_LABELS) or {
        case.get("label") for case in CLASSIFIER_CASES
    } != set(ROUTE_LABELS):
        raise ValueError("Configure exactly one labelled classifier case per route type.")
    return records


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    print("| " + " | ".join(cell(value) for value in headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(cell(value) for value in row) + " |")


def _metric_rows(metrics):
    return [
        (
            pattern,
            f"{_mean([row['score'] for row in rows]):.2f}",
            f"{_mean([row['tokens'] for row in rows]):.1f}",
            f"{_mean([row['latency'] for row in rows]):.3f}",
        )
        for pattern, rows in metrics.items()
    ]


def _judge(judge_llm, query, known, answer):
    score, reason, tokens, latency, raw = judge(judge_llm, query, known, answer)
    return {
        "score": score,
        "reason": reason,
        "judge_tokens": tokens,
        "judge_latency": latency,
        "judge_raw": raw,
    }


def _benchmark_split(records):
    shuffled = list(records)
    random.Random(SEED).shuffle(shuffled)
    benchmark = shuffled[:BENCHMARK_SIZE]
    examples = shuffled[BENCHMARK_SIZE:BENCHMARK_SIZE + FEW_SHOT_COUNT]
    if {row["query"] for row in benchmark} & {row["query"] for row in examples}:
        raise ValueError("Few-shot examples overlap the benchmark queries.")
    return benchmark, examples


def _run_part_a(generator, judge_llm, benchmark, examples):
    functions = {
        "Zero-shot": lambda query: zero_shot(generator, query),
        "Few-shot": lambda query: few_shot(generator, query, examples),
        "Instruction": lambda query: instruction(generator, query),
        "Self-critique": lambda query: self_critique(generator, query, return_details=True),
        "Structured": lambda query: structured(
            generator,
            query,
            required_fields=STRUCTURED_FIELDS,
            return_details=True,
        ),
    }
    metrics = {name: [] for name in PATTERN_NAMES}
    details = {}
    for item in benchmark:
        query = item["query"]
        for name, function in functions.items():
            result = function(query)
            answer, tokens, latency = result[:3]
            pattern_detail = result[3] if len(result) > 3 else None
            judged = _judge(judge_llm, query, item["answer"], answer)
            row = {
                "query": query,
                "known": item["answer"],
                "answer": answer,
                "tokens": tokens,
                "latency": latency,
                **judged,
            }
            if pattern_detail is not None:
                row["detail"] = pattern_detail
            metrics[name].append(row)
            details[(query, name)] = row
    return metrics, details


class _ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, prompt, temperature=0.0):
        return next(self.outputs), 1, 0.0


def _print_a3_retry_demo():
    malformed = '{"answer": "quantization'
    corrected = json.dumps(
        {
            "answer": "Reduces storage and arithmetic cost by lowering numeric precision.",
            "source_doc": "controlled parser-recovery demonstration",
            "type": "procedural",
        }
    )
    demo = _ScriptedModel([malformed, corrected])
    answer, tokens, latency, details = structured( # type: ignore
        demo,
        "Demonstrate parser recovery.",
        required_fields=STRUCTURED_FIELDS,
        return_details=True,
    )
    print(
        "Controlled parser-recovery test (scripted malformed response, not a model "
        "benchmark result):"
    )
    _table(
        ("Initial invalid response", "Validation error", "Retry response", "Parsed output"),
        (
            (
                malformed,
                details["initial_error"],
                details["raw"],
                answer,
            ),
        ),
    )
    print(f"Recovery-test calls: 2; tokens: {tokens}; latency: {latency:.3f}s.")


def _run_part_b(generator, judge_llm):
    rows = defaultdict(list)
    details = {}
    for problem_index, problem in enumerate(REASONING_PROBLEMS, start=1):
        query, known = problem["query"], problem["answer"]
        zero, zero_tokens, zero_latency = zero_shot(generator, query)
        zero_score = _judge(judge_llm, query, known, zero)
        rows["Zero-shot"].append(
            {"score": zero_score["score"], "tokens": zero_tokens, "latency": zero_latency}
        )

        cot, cot_tokens, cot_latency = chain_of_thought(generator, query)
        cot_final = final_answer(cot)
        cot_score = _judge(judge_llm, query, known, cot_final)
        rows["Chain-of-thought"].append(
            {"score": cot_score["score"], "tokens": cot_tokens, "latency": cot_latency}
        )

        vote, vote_tokens, vote_latency, samples = self_consistency( # type: ignore
            generator, query, return_details=True
        )
        vote_score = _judge(judge_llm, query, known, vote)
        rows["Self-consistency"].append(
            {"score": vote_score["score"], "tokens": vote_tokens, "latency": vote_latency}
        )

        verified, verified_tokens, verified_latency, plan_details = plan_solve_verify( # type: ignore
            generator, query, return_details=True
        )
        verified_score = _judge(judge_llm, query, known, verified)
        rows["Plan-Solve-Verify"].append(
            {
                "score": verified_score["score"],
                "tokens": verified_tokens,
                "latency": verified_latency,
            }
        )
        details[problem_index] = {
            "problem": query,
            "known": known,
            "zero_shot": zero,
            "chain_of_thought": cot,
            "chain_of_thought_final": cot_final,
            "samples": samples,
            "vote": vote,
            "plan": plan_details[0],
            "solution": plan_details[1],
            "critique": plan_details[2],
            "verified": plan_details[3],
        }

    react_answer, react_tokens, react_latency, trace = react( # type: ignore
        generator, REACT_QUERY, min_tool_calls=2, return_details=True
    )
    react_score = _judge(
        judge_llm,
        REACT_QUERY,
        REASONING_PROBLEMS[0]["answer"],
        final_answer(react_answer),
    )
    return rows, details, {
        "answer": react_answer,
        "tokens": react_tokens,
        "latency": react_latency,
        "trace": trace,
        "score": react_score["score"],
        "judge_reason": react_score["reason"],
        "query": REACT_QUERY,
    }


def _run_classifier_cases(generator):
    cases = []
    for case in CLASSIFIER_CASES:
        label, tokens, latency, prompt = classify(generator, case["query"], version=1)
        cases.append(
            {
                **case,
                "predicted": label,
                "correct": label == case["label"],
                "tokens": tokens,
                "latency": latency,
                "prompt": prompt,
            }
        )
    first_accuracy = sum(case["correct"] for case in cases)
    selected_version = 1
    if first_accuracy < 4:
        selected_version = 2
        for case in cases:
            label, tokens, latency, prompt = classify(
                generator, case["query"], version=selected_version
            )
            case.update(
                predicted_v2=label,
                correct_v2=label == case["label"],
                v2_tokens=tokens,
                v2_latency=latency,
                prompt_v2=prompt,
            )
    return cases, selected_version


def _run_part_c(generator, judge_llm, benchmark, zero_shot_rows, classifier_version):
    rows = []
    for item in benchmark:
        answer, label, tokens, latency = route(
            generator, item["query"], classifier_version=classifier_version
        )
        judged = _judge(judge_llm, item["query"], item["answer"], answer)
        rows.append(
            {
                "query": item["query"],
                "route": label,
                "answer": answer,
                "known": item["answer"],
                "score": judged["score"],
                "reason": judged["reason"],
                "tokens": tokens,
                "latency": latency,
            }
        )
    zero_rows = zero_shot_rows["Zero-shot"]
    counts = Counter(row["route"] for row in rows)
    most_common = counts.most_common()
    most_used = most_common[0][0]
    router_score = _mean([row["score"] for row in rows])
    zero_score = _mean([row["score"] for row in zero_rows])
    extra_tokens = _mean([row["tokens"] for row in rows]) - _mean(
        [row["tokens"] for row in zero_rows]
    )
    return {
        "rows": rows,
        "counts": counts,
        "most_used": most_used,
        "router_score": router_score,
        "zero_score": zero_score,
        "extra_tokens": extra_tokens,
    }


def _word_count(text):
    return len(text.split())


def run_assignment():
    records = validate_setup()
    from src.model import LLM

    print(f"Dataset: {DATASET_PATH} ({len(records)} validated examples)")
    print(f"Generator: {GENERATOR_MODEL}; PEFT adapter: {ADAPTER_PATH}")
    print(f"Judge: {JUDGE_MODEL}")
    generator = LLM(GENERATOR_MODEL, adapter_path=ADAPTER_PATH)
    judge_llm = LLM(JUDGE_MODEL, adapter_path=None) # type: ignore
    benchmark, examples = _benchmark_split(records)

    print("\n# Part A — Base prompts and judge benchmark")
    print("\nFew-shot examples (all outside the benchmark set):")
    _table(
        ("Example query", "Known answer"),
        [(item["query"], item["answer"]) for item in examples],
    )
    print("\nInstruction template:")
    print(
        "You are a precise domain assistant. Follow these steps: 1. Identify exactly "
        "what the question asks. 2. Use only relevant facts and state uncertainty "
        "when needed. 3. Give a concise, direct answer."
    )
    print(f"\nJudge model: `{JUDGE_MODEL}`")
    print("Judge prompt:")
    print("```text\n" + JUDGE_PROMPT_TEMPLATE + "\n```")

    part_a, a_details = _run_part_a(generator, judge_llm, benchmark, examples)
    print("\nA4 benchmark (pattern-call tokens and latency; judge calls excluded):")
    _table(
        ("Pattern", "Average score (0–2)", "Average tokens", "Average latency (s)"),
        _metric_rows(part_a),
    )
    print("\nFive zero-shot judge scores for manual spot-check:")
    _table(
        ("Query", "Known answer", "Candidate", "Judge score/reason", "Your agreement?"),
        [
            (
                row["query"],
                row["known"],
                row["answer"],
                f"{row['score']}: {row['reason']}",
                "PENDING MANUAL REVIEW",
            )
            for row in part_a["Zero-shot"][:5]
        ],
    )
    critique_score = _mean([row["score"] for row in part_a["Self-critique"]])
    baseline_score = _mean([row["score"] for row in part_a["Zero-shot"]])
    critique_tokens = _mean([row["tokens"] for row in part_a["Self-critique"]])
    baseline_tokens = _mean([row["tokens"] for row in part_a["Zero-shot"]])
    if critique_score > baseline_score:
        critique_verdict = (
            "The score gain justifies the extra tokens when answer quality is the "
            "priority."
        )
    elif critique_score == baseline_score:
        critique_verdict = (
            "The extra tokens are not justified by a score gain on this benchmark."
        )
    else:
        critique_verdict = (
            "The extra tokens are not justified because self-critique scored lower."
        )
    print(
        f"Self-critique averaged {critique_score:.2f} versus {baseline_score:.2f} "
        f"for zero-shot, using {critique_tokens - baseline_tokens:.1f} additional "
        f"tokens per query. {critique_verdict}"
    )
    print("\nA2 self-critique traces (first two benchmark queries):")
    for query in [item["query"] for item in benchmark[:2]]:
        first, critique, revised = a_details[(query, "Self-critique")]["detail"]
        first_judgment = _judge(
            judge_llm, query, a_details[(query, "Self-critique")]["known"], first
        )
        revised_score = a_details[(query, "Self-critique")]["score"]
        print(f"\nQuery: {query}\nFirst answer: {first}\nCritique: {critique}\nRevised answer: {revised}")
        if revised_score > first_judgment["score"]:
            verdict = "the revision improved the judge score."
        elif revised_score == first_judgment["score"]:
            verdict = "the judge score was unchanged by the revision."
        else:
            verdict = "the revision lowered the judge score."
        print(
            f"First/revised judge scores: {first_judgment['score']}/{revised_score}; "
            f"{verdict}"
        )
    structured_rows = part_a["Structured"]
    valid_outputs = [
        row for row in structured_rows if "detail" in row
    ][:3]
    print("\nA3 structured schema:")
    print(json.dumps({key: "string" for key in STRUCTURED_FIELDS}, indent=2))
    print("Three valid model outputs:")
    _table(
        ("Query", "Parsed JSON", "Retry used?"),
        [
            (
                row["query"],
                row["answer"],
                row["detail"]["initial_error"] is not None,
            )
            for row in valid_outputs
        ],
    )
    if len(valid_outputs) < 3:
        print(
            f"Only {len(valid_outputs)} benchmark outputs were valid on the first "
            "attempt; successful retries are listed below."
        )
    retries = [
        row for row in structured_rows
        if "detail" in row and row["detail"]["initial_error"] is not None
    ]
    if retries:
        print("Actual benchmark parse failure and model retry:")
        _table(
            ("Query", "Initial error", "Retry output"),
            [
                (row["query"], row["detail"]["initial_error"], row["detail"]["raw"])
                for row in retries[:1]
            ],
        )
    else:
        _print_a3_retry_demo()

    print("\n# Part B — Reasoning, ReAct and plan/solve/verify")
    part_b, b_details, react_result = _run_part_b(generator, judge_llm)
    print("\nB1/B3 judge results:")
    _table(
        ("Pattern", "Problem 1 score", "Problem 2 score", "Average tokens"),
        [
            (
                name,
                part_b[name][0]["score"],
                part_b[name][1]["score"],
                f"{_mean([row['tokens'] for row in part_b[name]]):.1f}",
            )
            for name in ("Zero-shot", "Chain-of-thought", "Self-consistency", "Plan-Solve-Verify")
        ],
    )
    for index, detail in b_details.items():
        print(f"\nProblem {index}: {detail['problem']}")
        print(f"Known answer: {detail['known']}")
        print(f"Chain-of-thought reasoning and final answer:\n{detail['chain_of_thought']}")
        print("Five self-consistency final answers:")
        for sample_index, sample in enumerate(detail["samples"], start=1):
            print(f"  {sample_index}. {sample}")
        print(f"Majority vote (earliest answer in a tied largest group): {detail['vote']}")
    first_problem = b_details[1]
    print("\nB3 plan/solve/verify trace for Problem 1:")
    print(
        f"Plan: {first_problem['plan']}\n"
        f"Solution: {first_problem['solution']}\n"
        f"Verification critique: {first_problem['critique']}\n"
        f"Verified answer: {first_problem['verified']}"
    )
    single_patterns = ("Zero-shot", "Chain-of-thought", "Self-consistency")
    best_single = max(
        single_patterns,
        key=lambda name: _mean([row["score"] for row in part_b[name]]),
    )
    chain_score = _mean([row["score"] for row in part_b["Plan-Solve-Verify"]])
    best_score = _mean([row["score"] for row in part_b[best_single]])
    chain_tokens = _mean([row["tokens"] for row in part_b["Plan-Solve-Verify"]])
    best_tokens = _mean([row["tokens"] for row in part_b[best_single]])
    comparison = (
        f"Plan-Solve-Verify scored {chain_score:.2f} on average versus "
        f"{best_score:.2f} for the best single pattern ({best_single})."
    )
    if chain_score > best_score:
        comparison += (
            f" It used {chain_tokens - best_tokens:.1f} additional average tokens; "
            "the quality gain should be weighed against that cost for high-stakes use."
        )
    elif chain_score == best_score:
        comparison += (
            f" It used {chain_tokens - best_tokens:.1f} additional average tokens "
            "without a score gain, so this run does not justify the extra cost for "
            "ordinary queries."
        )
    else:
        comparison += (
            f" It used {chain_tokens - best_tokens:.1f} additional average tokens "
            "and scored lower, so this run does not justify its extra cost."
        )
    print(comparison)

    print("\nB2 ReAct query, full trace, and final answer:")
    print(react_result["query"])
    for index, step in enumerate(react_result["trace"], start=1):
        print(f"Trace {index}: {step}")
    print(f"Final answer: {final_answer(react_result['answer'])}")
    print(
        f"Judge score: {react_result['score']} ({react_result['judge_reason']}); "
        f"pattern tokens: {react_result['tokens']}; "
        f"latency: {react_result['latency']:.3f}s."
    )
    print(
        f"Model cycles: {sum(not step.startswith('Observation: ') for step in react_result['trace'])}; "
        f"minimum three cycles met: "
        f"{sum(not step.startswith('Observation: ') for step in react_result['trace']) >= 3}."
    )

    print("\n# Part C — Query router")
    classifier_cases, classifier_version = _run_classifier_cases(generator)
    print("\nC1 classifier prompt, version 1:")
    print("```text\n" + CLASSIFIER_PROMPT_V1 + "\n```")
    print("Version 1 labelled-query results:")
    _table(
        ("Hand label", "Query", "Predicted", "Correct?"),
        [
            (case["label"], case["query"], case["predicted"], case["correct"])
            for case in classifier_cases
        ],
    )
    version1_correct = sum(case["correct"] for case in classifier_cases)
    print(f"Version 1 accuracy: {version1_correct}/5.")
    if classifier_version == 2:
        print("\nAccuracy was below 4/5; refined classifier prompt:")
        print("```text\n" + CLASSIFIER_PROMPT_V2 + "\n```")
        _table(
            ("Hand label", "Version 2 prediction", "Correct?"),
            [
                (case["label"], case["predicted_v2"], case["correct_v2"])
                for case in classifier_cases
            ],
        )
        print(
            f"Version 2 accuracy: "
            f"{sum(case['correct_v2'] for case in classifier_cases)}/5."
        )
    router_results = _run_part_c(
        generator, judge_llm, benchmark, part_a, classifier_version
    )
    print("\nC2 router results on the A4 benchmark:")
    _table(
        ("Query", "Route chosen", "Judge score", "Tokens (classifier included)"),
        [
            (row["query"], row["route"], row["score"], row["tokens"])
            for row in router_results["rows"]
        ],
    )
    improved = [
        (index, row)
        for index, row in enumerate(router_results["rows"])
        if row["score"] > part_a["Zero-shot"][index]["score"]
    ]
    print("\nQueries where the router scored higher than zero-shot:")
    _table(
        ("Query", "Zero-shot score", "Router score", "Router tokens"),
        [
            (
                row["query"],
                part_a["Zero-shot"][index]["score"],
                row["score"],
                row["tokens"],
            )
            for index, row in improved
        ],
    )
    print("\nC3 cost/quality comparison (judge-call cost excluded for both rows):")
    zero_rows = part_a["Zero-shot"]
    routed_rows = router_results["rows"]
    _table(
        ("Approach", "Average score (0–2)", "Average tokens", "Average latency (s)"),
        (
            (
                "Always zero-shot",
                f"{_mean([row['score'] for row in zero_rows]):.2f}",
                f"{_mean([row['tokens'] for row in zero_rows]):.1f}",
                f"{_mean([row['latency'] for row in zero_rows]):.3f}",
            ),
            (
                "Router",
                f"{_mean([row['score'] for row in routed_rows]):.2f}",
                f"{_mean([row['tokens'] for row in routed_rows]):.1f}",
                f"{_mean([row['latency'] for row in routed_rows]):.3f}",
            ),
        ),
    )
    if improved:
        improved_indices = [index for index, _ in improved]
        improved_token_delta = _mean(
            [routed_rows[index]["tokens"] - zero_rows[index]["tokens"] for index in improved_indices]
        )
        improved_score_delta = _mean(
            [routed_rows[index]["score"] - zero_rows[index]["score"] for index in improved_indices]
        )
        if improved_token_delta > 0:
            cost_verdict = (
                f"On those queries I consider the extra {improved_token_delta:.1f} "
                f"tokens worthwhile for the {improved_score_delta:.2f}-point score gain."
            )
        else:
            cost_verdict = (
                f"Those queries gained {improved_score_delta:.2f} score points without "
                "an average token increase."
            )
    else:
        cost_verdict = (
            "The extra classifier cost was not worthwhile here because no routed "
            "query scored higher."
        )
    per_route = ", ".join(f"{key}: {value}" for key, value in router_results["counts"].items())
    report = (
        f"The router selected {router_results['most_used']} most often "
        f"({router_results['counts'][router_results['most_used']]}/10). It beat "
        f"zero-shot on {len(improved)} queries; see the query list above for the exact "
        f"questions. {cost_verdict} Average scores were "
        f"{_mean([row['score'] for row in zero_rows]):.2f} for zero-shot and "
        f"{_mean([row['score'] for row in routed_rows]):.2f} for routing. Router "
        f"tokens include classification and averaged "
        f"{_mean([row['tokens'] for row in routed_rows]) - _mean([row['tokens'] for row in zero_rows]):.1f} "
        "more per query; judge calls are excluded for both approaches. "
        f"{max(PATTERN_NAMES, key=lambda name: _mean([row['score'] for row in part_a[name]]))} "
        "had the top Part A average score, so I would deploy it if routing were "
        "unavailable. This fixed ten-query result is directional, not conclusive; "
        "validate on a larger domain-specific set before production."
    )
    print("\nC3 analysis (~100 words):")
    print(report)
    print(f"Word count: {_word_count(report)}")
    print(
        "\nAssignment run complete. Review the five A4 spot-checks yourself, record "
        "how many you agree with, then export this notebook to HTML."
    )
    return {
        "benchmark": benchmark,
        "few_shot_examples": examples,
        "part_a": part_a,
        "part_b": part_b,
        "part_b_details": b_details,
        "react": react_result,
        "classifier_cases": classifier_cases,
        "classifier_version": classifier_version,
        "router": router_results,
    }


if __name__ == "__main__":
    run_assignment()
