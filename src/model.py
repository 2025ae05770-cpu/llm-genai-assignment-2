import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ADAPTER_PATH, LOAD_IN_4BIT, MAX_NEW_TOKENS


class LLM:
    def __init__(
        self, model_name, adapter_path=ADAPTER_PATH, load_in_4bit=LOAD_IN_4BIT
    ):
        if not model_name:
            raise ValueError("Set both model IDs in config.py before running.")
        if adapter_path and not Path(adapter_path).is_dir():
            raise FileNotFoundError(
                f"PEFT adapter not found: {adapter_path}. Copy Assignment 1's "
                "qlora_adapter_B directory there, or set ADAPTER_PATH = None to "
                "intentionally benchmark the unadapted quantized base model."
            )
        if load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit bitsandbytes loading requires a supported CUDA environment. "
                "Run on the Assignment 1 GPU lab, or explicitly set LOAD_IN_4BIT = False "
                "on hardware with enough memory."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, temperature=0.0):
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        latency = time.perf_counter() - start
        generated = output[0, inputs.input_ids.shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        tokens = int(inputs.input_ids.numel() + generated.numel())
        return answer, tokens, latency
