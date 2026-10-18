# Assignment 2A — Prompting as System Design

This project implements the assignment's prompting patterns, reasoning chains, LLM
judge, ReAct calculator, classifier and query router. The submission notebook is
`assignment2.ipynb`; run it from the project root so the notebook captures the real
model outputs and benchmark tables, then export that same notebook to HTML.

## Assignment 1 artifacts and model setup

The supplied Assignment 1 files are in `Assignment-1/`. The 59-example
`instruction_dataset.jsonl` is read directly from there; dataset rows with
`instruction` and `response` are mapped to the query/known-answer fields required
here.

Assignment 1 identifies the generator base as `Qwen/Qwen2.5-7B` and records a
4-bit NF4 QLoRA run with Adapter B (`r=16`, `alpha=32`, `q_proj` and `v_proj`).
The notebook, HTML report, dataset and domain corpus are present, but the saved
`qlora_adapter_B` model directory is **not** included. The default configuration
expects the adapter at `Assignment-1/qlora_adapter_B`; copy that directory from the
Assignment 1 training environment before running if you want to benchmark the
fine-tuned model. Do not treat the notebook's logged training outputs as the model
weights.

Configuration is in `config.py`. The default judge is `Qwen/Qwen2.5-7B-Instruct`,
which differs from the base generator. The default 4-bit NF4 loading matches the
Assignment 1 setup and requires a supported CUDA environment and `bitsandbytes`.
The original Assignment 1 notebook reports an A100 GPU environment. For a base-only
run, explicitly set `ADAPTER_PATH = None`; record that choice in the submission
because it does not evaluate the fine-tuned adapter. Do not fabricate scores when a
model or adapter is unavailable.

## Run on the university CUDA machine

The commands below assume the university machine runs Linux. Run them in a terminal
on the GPU machine, or in the university's CUDA-enabled Jupyter environment. If the
university supplies a preconfigured course environment, use it rather than creating
a second environment or changing its CUDA/PyTorch versions.

### 1. Get the complete project and check out a GPU

Copy the project to your university account, including the `Assignment-1` folder.
For example, if using `scp` from your own computer, run this from the computer that
has the project (replace the host and destination with the university's instructions):

```bash
scp -r LLM-GenAI-Assignment-2 YOUR_USER@YOUR_GPU_HOST:~/LLM-GenAI-Assignment-2
```

Log in to the university GPU host using its approved method. If the university uses
a scheduler, request an interactive GPU session before continuing. Check that a GPU
has been assigned:

```bash
nvidia-smi
```

### 2. Enter the project and verify the Assignment 1 files

```bash
cd ~/LLM-GenAI-Assignment-2
test -f Assignment-1/instruction_dataset.jsonl && echo "Dataset found"
test -d Assignment-1/qlora_adapter_B && echo "Adapter found"
```

The dataset should already be present. The second command should also succeed after
you copy the trained `qlora_adapter_B` directory from the Assignment 1 training
environment. The folder should contain the PEFT adapter configuration and weight
file (for example, `adapter_config.json` and `adapter_model.safetensors`). The
notebook or HTML report is not a substitute for these weights.

### 3. Activate a CUDA-compatible Python environment

If your course/GPU environment has an activation command, activate it now and skip
the optional virtual environment creation below. Otherwise, create an environment
with the Python version supported by the university's PyTorch/CUDA setup (Python
3.10 or newer is recommended):

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Check that PyTorch can see the assigned GPU **before** installing other packages:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

`CUDA available` must be `True`. If PyTorch is missing, reports `False`, or its
CUDA build is incompatible with the host, use the PyTorch install command/version
recommended by the university first. Do not replace a working university CUDA
PyTorch build with an arbitrary `pip install torch`.

### 4. Install the project packages

With the intended Python environment active, install the packages:

```bash
python -m pip install -r requirements.txt
```

The requirements include PyTorch and `bitsandbytes`. After installation, repeat
the CUDA check above. If installing `requirements.txt` changed PyTorch to a
non-CUDA build, restore the university-approved CUDA PyTorch version before
continuing. Resolve any `bitsandbytes`/CUDA compatibility issues using the
university's supported versions; do not proceed with a CPU run and report its
metrics as the CUDA benchmark.

### 5. Verify configuration and model access

The default [config.py](./config.py) uses the Assignment 1 base model
`Qwen/Qwen2.5-7B`, the local adapter at `Assignment-1/qlora_adapter_B`, and the
different judge model `Qwen/Qwen2.5-7B-Instruct`. Both are loaded in 4-bit NF4.
The first run downloads model files from Hugging Face, so the machine needs network
access to the model host or an approved local model cache with sufficient storage.

Check the adapter configuration and dataset:

```bash
test -f Assignment-1/qlora_adapter_B/adapter_config.json && echo "Adapter config found"
python -c "from assignment2 import validate_setup; print('Validated dataset rows:', len(validate_setup()))"
```

If adapter weights are stored somewhere else, update `ADAPTER_PATH` in `config.py`
to that directory. Keep `ADAPTER_PATH` set to the trained adapter to evaluate the
Assignment 1 fine-tuned model. Setting it to `None` deliberately evaluates the
unadapted 4-bit base model instead; disclose that difference in your submission.

### 6. Run all notebook cells

Start Jupyter from the project root so Python can import `config.py` and `src/`:

```bash
python -m jupyter lab
```

Open `assignment2.ipynb` in JupyterLab (or connect VS Code to the university
machine and open the notebook there). Select the Python environment checked above,
then choose **Run All Cells**. The first cell validates the setup; the next runs the
full benchmark for Parts A–C. Model downloads and evaluation can take a while. Keep
the GPU allocation active until the benchmark finishes and save the notebook after
the outputs appear.

Review the five A4 judge spot-checks printed by the benchmark. In the final manual
review cell, replace the five `None` values with `True` or `False` to record whether
you agree with each score, then run that cell and save the notebook again.

### 7. Export the executed notebook to HTML

In a terminal with the project environment active, and from the project root, run:

```bash
python -m jupyter nbconvert --to html assignment2.ipynb
```

This creates `assignment2.html` beside the notebook. Open it and verify that it
contains the executed outputs, tables, manual spot-check count and no setup errors.
Submit the executed `assignment2.ipynb` and its matching `assignment2.html`.

If the notebook stops during setup, fix the reported dataset, adapter, model access
or CUDA issue and rerun it. Do not submit placeholder cells or mocked outputs as
real benchmark results.

The run covers A1-A4, B1-B3 and C1-C3, including pattern and router metrics. The
two reasoning problems and five manually labelled classifier examples are defined
in `config.py` and are specific to the computer-vision/edge-AI domain in the
Assignment 1 dataset. The structured-output retry test is clearly labelled as a
controlled parser-recovery demonstration if none of the actual benchmark outputs
needs a retry.

## Local checks

Pattern and parser logic can be tested without downloading the large models:

```powershell
python -m compileall assignment2.py config.py src
```

The full benchmark requires the configured model files, the Assignment 1 adapter
(unless explicitly disabled), and suitable GPU resources.
