import os
import threading
import time
import json
import csv
from dataclasses import dataclass, asdict
from typing import Optional, List
from textwrap import dedent
import torch
import psutil

# Import your updated Engine
from engine.engine import Engine


@dataclass
class BenchmarkResult:
    prompt_length: int
    output_length: int
    generated_tokens: int
    ttft: float        # seconds
    total_time: float  # seconds
    tokens_per_second: float
    per_token_latency: float
    peak_memory_mb: float
    cpu_usage_percent: float
    error: Optional[str] = None

    def __str__(self):
        if self.error:
            return f"❌ FAILED: {self.error}"
        return dedent(f"""\
            Benchmark Results:
            ==================
            Input tokens:     {self.prompt_length}
            Output tokens:    {self.output_length} (gen: {self.generated_tokens})
            TTFT:             {self.ttft * 1000:6.2f} ms
            Total time:       {self.total_time:6.3f} s
            Throughput:       {self.tokens_per_second:6.2f} tok/s
            Per-token (after first): {self.per_token_latency * 1000:5.2f} ms
            Peak GPU memory:  {self.peak_memory_mb:7.1f} MB
            Avg CPU usage:    {self.cpu_usage_percent:6.1f} %
            """).strip()


class Benchmark:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.process = psutil.Process(os.getpid())

    def _reset_memory_stats(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def _get_peak_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0

    def _monitor_cpu_percent(self, duration: float, interval: float = 0.01) -> float:
        """Sample CPU% over `duration` seconds and return average"""
        samples = []
        start = time.time()
        # Warm-up
        self.process.cpu_percent(interval=0.01)
        while time.time() - start < duration:
            samples.append(self.process.cpu_percent(interval=None))
            time.sleep(interval)
        return sum(samples) / len(samples) if samples else 0.0

    def run_single_benchmark(
        self,
        prompt: str,
        warmup: bool = False,
        max_new_tokens: Optional[int] = None,
    ) -> Optional[BenchmarkResult]:
        try:
            self._reset_memory_stats()

            # Tokenize to get prompt length
            inputs = self.engine.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - (max_new_tokens or 128),
            )
            prompt_len = inputs["input_ids"].shape[1]

            # Override max_new_tokens if specified
            original_max = self.engine.params.max_new_tokens
            if max_new_tokens is not None:
                self.engine.params.max_new_tokens = max_new_tokens

            start_time = time.perf_counter()
            ttft = None
            token_count = 0

            # Use new streaming API
            stream = self.engine.generate_stream(prompt)

            # Wait for first token → accurate TTFT
            try:
                first_token = next(stream)
                ttft = time.perf_counter() - start_time
                token_count = 1
            except StopIteration:
                ttft = time.perf_counter() - start_time
                token_count = 0

            # Collect remaining tokens
            for _ in stream:
                token_count += 1

            total_time = time.perf_counter() - start_time

            # Restore original param
            self.engine.params.max_new_tokens = original_max

            # Metrics
            output_len = prompt_len + token_count
            tokens_per_sec = token_count / total_time if total_time > 0 else 0.0
            gen_time = total_time - (ttft or 0.0)
            per_token_lat = gen_time / max(token_count - 1, 1) if token_count > 1 else 0.0

            # System stats (CPU sampled over gen duration)
            cpu_usage = self._monitor_cpu_percent(total_time)
            peak_mem = self._get_peak_memory_mb()

            if warmup:
                return None

            return BenchmarkResult(
                prompt_length=prompt_len,
                output_length=output_len,
                generated_tokens=token_count,
                ttft=ttft,
                total_time=total_time,
                tokens_per_second=tokens_per_sec,
                per_token_latency=per_token_lat,
                peak_memory_mb=peak_mem,
                cpu_usage_percent=cpu_usage,
                error=None,
            )

        except Exception as e:
            if not warmup:
                return BenchmarkResult(
                    prompt_length=0, output_length=0, generated_tokens=0,
                    ttft=0.0, total_time=0.0, tokens_per_second=0.0,
                    per_token_latency=0.0, peak_memory_mb=0.0,
                    cpu_usage_percent=0.0, error=str(e)
                )
            return None

    def run_benchmark_suite(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        num_warmup: int = 2,
        num_iterations: int = 5,
    ) -> List[BenchmarkResult]:
        if prompts is None:
            prompts = [
                "Hello!",
                "Explain how attention works in transformers.",
                "Write a Python function to compute Fibonacci numbers recursively and iteratively. Discuss time complexity.",
            ]

        all_results = []

        for i, prompt in enumerate(prompts):
            print(f"\n{'='*70}")
            print(f"🧪 Prompt {i+1}/{len(prompts)} | max_new_tokens={max_new_tokens}")
            print(f"   Preview: {repr(prompt[:60])}{'...' if len(prompt)>60 else ''}")
            print("="*70)

            # Warmup
            for w in range(num_warmup):
                self.run_single_benchmark(prompt, warmup=True, max_new_tokens=max_new_tokens)
            print(f"✅ {num_warmup} warmup runs completed.")

            # Benchmark
            results = []
            for it in range(num_iterations):
                res = self.run_single_benchmark(
                    prompt, warmup=False, max_new_tokens=max_new_tokens
                )
                if res and not res.error:
                    results.append(res)
                    print(
                        f"   [{it+1}/{num_iterations}] "
                        f"TTFT: {res.ttft*1000:5.1f}ms | "
                        f"Speed: {res.tokens_per_second:5.1f} tok/s | "
                        f"Mem: {res.peak_memory_mb:5.0f}MB"
                    )
                else:
                    err = res.error if res else "Unknown"
                    print(f"   [{it+1}/{num_iterations}] ❌ {err}")

            # Aggregate
            if results:
                avg = self._average_results(results)
                all_results.append(avg)
                print("\n📊 Average:")
                print(avg)
            else:
                dummy = BenchmarkResult(0,0,0,0,0,0,0,0,0,error="All failed")
                all_results.append(dummy)

        return all_results

    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        n = len(results)
        return BenchmarkResult(
            prompt_length=round(sum(r.prompt_length for r in results) / n),
            output_length=round(sum(r.output_length for r in results) / n),
            generated_tokens=round(sum(r.generated_tokens for r in results) / n),
            ttft=sum(r.ttft for r in results) / n,
            total_time=sum(r.total_time for r in results) / n,
            tokens_per_second=sum(r.tokens_per_second for r in results) / n,
            per_token_latency=sum(r.per_token_latency for r in results) / n,
            peak_memory_mb=sum(r.peak_memory_mb for r in results) / n,
            cpu_usage_percent=sum(r.cpu_usage_percent for r in results) / n,
            error=None,
        )

    def save_json(self, results: List[BenchmarkResult], path: str = "naive_no_cache.json"):
        data = [asdict(r) for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Saved to {path}")


if __name__ == "__main__":
    engine = Engine()
    engine.load_model("Qwen/Qwen2.5-0.5B-Instruct")

    bench = Benchmark(engine)
    results = bench.run_benchmark_suite(
        num_warmup=2,
        num_iterations=3,
        max_new_tokens=1024,
    )

    bench.save_json(results)