from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Assignment-1" / "instruction_dataset.jsonl"

GENERATOR_MODEL = "Qwen/Qwen2.5-7B"
ADAPTER_PATH = PROJECT_DIR / "Assignment-1" / "qlora_adapter_B"
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOAD_IN_4BIT = True

MAX_NEW_TOKENS = 200
BENCHMARK_SIZE = 10
FEW_SHOT_COUNT = 3
SEED = 42

STRUCTURED_FIELDS = ("answer", "source_doc", "type")

REASONING_PROBLEMS = (
    {
        "query": (
            "An edge vision detector must process 30 frames per second. Its inference "
            "takes 24 milliseconds per frame. What is the per-frame latency budget, "
            "how much latency remains, and does it meet the real-time requirement?"
        ),
        "answer": (
            "The budget is 1000/30 = 33.33 ms per frame. The detector has "
            "33.33 - 24 = 9.33 ms of spare latency, so it meets the requirement."
        ),
    },
    {
        "query": (
            "An edge computer-vision model has 240 MB of 32-bit weights. If the "
            "weights are quantized to 8 bits and storage scales proportionally with "
            "bit width, what is the new size and how much storage is saved in MB "
            "and percent?"
        ),
        "answer": (
            "The size becomes 240 * (8/32) = 60 MB. Storage saved is "
            "240 - 60 = 180 MB, which is 180/240 * 100 = 75%."
        ),
    },
)

REACT_QUERY = REASONING_PROBLEMS[0]["query"]

CLASSIFIER_CASES = (
    {
        "query": "What is overfitting, and why do CNNs need to guard against it?",
        "label": "SIMPLE_FACTUAL",
    },
    {
        "query": (
            "Extract the answer, source_doc, and type for this question as JSON: "
            "How does quantization reduce the computational cost of a neural network?"
        ),
        "label": "STRUCTURED_EXTRACTION",
    },
    {
        "query": (
            "An edge deployment needs object classes and precise object locations, "
            "while operating under a strict latency limit. Which computer-vision task "
            "provides object locations, and why might a faster single-stage detector "
            "be preferred to a two-stage detector?"
        ),
        "label": "MULTI_STEP_REASONING",
    },
    {"query": REACT_QUERY, "label": "NEEDS_TOOL"},
    {
        "query": (
            "A thermal camera failed to detect a person in a hazardous industrial "
            "area. What immediate safety steps should the site take, and what "
            "limitations of the detection system must be checked?"
        ),
        "label": "HIGH_STAKES",
    },
)
