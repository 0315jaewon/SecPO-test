"""lm-eval generation interface for a saved Tinker SecPO checkpoint."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lm_eval.api.model import LM

from .evaluation_common import CheckpointSampler, MODEL, TOKENIZER_REVISION, now, write_json


def set_task_output_cap(task, max_tokens: int) -> None:
    """Override task-level limits too, notably MMLU-Pro's explicit 2048."""
    if type(max_tokens) is not int or not 0 < max_tokens < 65536:
        raise ValueError("Output cap must be between 1 and 65535 tokens")
    kwargs = {**(task.get_config("generation_kwargs") or {}), "max_gen_toks": max_tokens}
    task.set_config("generation_kwargs", kwargs)


class TinkerEvaluationLM(LM):
    def __init__(self, sampler: CheckpointSampler, output_dir: Path, *, concurrency=12, default_max_tokens=2048):
        super().__init__()
        self.sampler = sampler
        self.tokenizer = sampler.tokenizer
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.default_max_tokens = default_max_tokens
        self.length_limited = 0

    @property
    def tokenizer_name(self):
        return MODEL + "@" + TOKENIZER_REVISION + ":thinking-off"

    def chat_template(self, chat_template=False):
        return self.tokenizer.chat_template

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False, preserve_thinking=False,
        )

    def loglikelihood(self, requests):
        raise NotImplementedError("This evaluation adapter supports generation tasks only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("This evaluation adapter supports generation tasks only")

    def generate_until(self, requests):
        outputs = [None] * len(requests)
        def generate(index):
            prompt, kwargs = requests[index].args
            kwargs = dict(kwargs)
            stops = kwargs.pop("until", [])
            if isinstance(stops, str):
                stops = [stops]
            max_tokens = kwargs.pop("max_gen_toks", self.default_max_tokens)
            temperature = kwargs.pop("temperature", 0.0)
            do_sample = kwargs.pop("do_sample", False)
            top_p = kwargs.pop("top_p", 1.0)
            top_k = kwargs.pop("top_k", -1)
            penalties = {name: kwargs.pop(name) for name in
                         ("min_p", "presence_penalty", "repetition_penalty", "frequency_penalty")
                         if name in kwargs}
            if kwargs:
                raise ValueError(f"Unhandled generation options: {sorted(kwargs)}")
            result = self.sampler.generate(prompt, max_tokens=max_tokens, stops=stops,
                                           temperature=temperature if do_sample else 0.0,
                                           top_p=top_p, top_k=top_k, **penalties)
            return index, result.text, result.stop_reason
        completed = 0
        truncated = 0
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(generate, i) for i in range(len(requests))]
            for future in as_completed(futures):
                index, text, reason = future.result()
                outputs[index] = text
                completed += 1
                truncated += reason == "length"
                self.length_limited = truncated
                write_json(self.output_dir / "progress.json", {
                    "status": "generating", "updated_at": now(), "completed": completed,
                    "total": len(requests), "length_limited": truncated, **self.sampler.stats(),
                })
                if completed % 25 == 0 or completed == len(requests):
                    print(f"Generated {completed}/{len(requests)}; length-limited={truncated}", flush=True)
        return outputs
