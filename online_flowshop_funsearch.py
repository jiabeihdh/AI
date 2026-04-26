import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Optional

import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
FUNSEARCH_DIR = PROJECT_DIR / "funsearch"
if str(FUNSEARCH_DIR) not in sys.path:
    sys.path.append(str(FUNSEARCH_DIR))

# You can fill these directly if you do not want to use environment variables.
HARDCODED_API_KEY = ""
HARDCODED_BASE_URL = "https://api.bltcy.ai/v1"
HARDCODED_MODEL_NAME = "gpt-4o-mini"
MAX_LLM_RETRIES = 5

from implementation import code_manipulation
from implementation import config
from implementation import evaluator
from implementation import funsearch
from implementation import profile
from implementation import programs_database
from implementation import sampler
from MoE import DynamicHeuristicFusion
from diversity import BehavioralFingerprinter


DEFAULT_SPECIFICATION = r'''
import numpy as np


def compute_flowshop_makespan(sequence: np.ndarray, processing_times: np.ndarray, release_times: np.ndarray) -> float:
    """Computes makespan of a permutation flow-shop sequence with release times."""
    num_jobs = len(sequence)
    if num_jobs == 0:
        return 0.0

    num_machines = processing_times.shape[1]
    completion = np.zeros(num_machines, dtype=float)
    for idx in range(num_jobs):
        job_id = int(sequence[idx])
        job_times = processing_times[job_id]
        completion[0] = max(completion[0], float(release_times[job_id])) + float(job_times[0])
        for machine_id in range(1, num_machines):
            completion[machine_id] = max(completion[machine_id], completion[machine_id - 1]) + float(job_times[machine_id])
    return float(completion[-1])


def compute_completion_vector(sequence: np.ndarray, processing_times: np.ndarray, release_times: np.ndarray) -> np.ndarray:
    """Computes machine completion times for the current sequence."""
    num_machines = processing_times.shape[1]
    completion = np.zeros(num_machines, dtype=float)
    for idx in range(len(sequence)):
        job_id = int(sequence[idx])
        job_times = processing_times[job_id]
        completion[0] = max(completion[0], float(release_times[job_id])) + float(job_times[0])
        for machine_id in range(1, num_machines):
            completion[machine_id] = max(completion[machine_id], completion[machine_id - 1]) + float(job_times[machine_id])
    return completion


def simulate_online_schedule(processing_times: np.ndarray, release_times: np.ndarray, arrival_order: np.ndarray) -> tuple[np.ndarray, float]:
    """Schedules jobs online by repeatedly calling the evolved heuristic."""
    num_jobs, num_machines = processing_times.shape
    current_sequence = np.zeros(0, dtype=int)
    machine_completion_times = np.zeros(num_machines, dtype=float)

    for step, job_id in enumerate(arrival_order):
        job_id = int(job_id)
        sequence_processing_times = (
            processing_times[current_sequence]
            if len(current_sequence) > 0 else np.zeros((0, num_machines), dtype=float)
        )
        sequence_release_times = (
            release_times[current_sequence]
            if len(current_sequence) > 0 else np.zeros(0, dtype=float)
        )

        insert_index = heuristic(
            current_sequence=current_sequence.copy(),
            job_processing_times=processing_times[job_id].copy(),
            machine_completion_times=machine_completion_times.copy(),
            sequence_processing_times=sequence_processing_times.copy(),
            current_job_release_time=float(release_times[job_id]),
            sequence_release_times=sequence_release_times.copy(),
        )

        if not isinstance(insert_index, (int, np.integer, float, np.floating)):
            insert_index = len(current_sequence)
        insert_index = int(insert_index)
        insert_index = max(0, min(insert_index, len(current_sequence)))

        new_sequence = np.empty(step + 1, dtype=int)
        if insert_index > 0:
            new_sequence[:insert_index] = current_sequence[:insert_index]
        new_sequence[insert_index] = job_id
        if insert_index < len(current_sequence):
            new_sequence[insert_index + 1:] = current_sequence[insert_index:]
        current_sequence = new_sequence
        machine_completion_times = compute_completion_vector(current_sequence, processing_times, release_times)

    final_makespan = compute_flowshop_makespan(current_sequence, processing_times, release_times)
    return current_sequence, final_makespan


@funsearch.run
def evaluate(instance: dict) -> float:
    """Returns a normalized score to maximize on a single online flow-shop instance."""
    processing_times = np.asarray(instance['processing_times'], dtype=float)
    release_times = np.asarray(instance['release_times'], dtype=float)
    arrival_order = np.asarray(instance['arrival_order'], dtype=int)
    baseline_makespan = float(instance['baseline_makespan'])

    _, candidate_makespan = simulate_online_schedule(processing_times, release_times, arrival_order)
    denom = max(baseline_makespan, 1e-9)
    relative_improvement = (baseline_makespan - candidate_makespan) / denom
    return float(relative_improvement)


@funsearch.evolve
def heuristic(
    current_sequence: np.ndarray,
    job_processing_times: np.ndarray,
    machine_completion_times: np.ndarray,
    sequence_processing_times: np.ndarray,
    current_job_release_time: float,
    sequence_release_times: np.ndarray,
) -> int:
    """Returns the insertion position for the newly arrived job."""
    seq_len = len(current_sequence)
    if seq_len == 0:
        return 0

    job_total = float(np.sum(job_processing_times))
    job_front = float(job_processing_times[0])
    job_back = float(job_processing_times[-1])
    current_tail = float(machine_completion_times[-1]) if len(machine_completion_times) > 0 else 0.0

    if sequence_processing_times.size == 0:
        return 0

    sequence_totals = np.sum(sequence_processing_times, axis=1)
    mean_total = float(np.mean(sequence_totals))
    mean_release = float(np.mean(sequence_release_times)) if len(sequence_release_times) > 0 else current_job_release_time

    if current_job_release_time > mean_release and job_total >= mean_total:
        return seq_len
    if current_job_release_time <= mean_release and job_front <= job_back and current_tail > current_job_release_time:
        return 0
    return seq_len // 2
'''


def _replace_heuristic_body(specification: str, new_body: str) -> str:
    start_token = '@funsearch.evolve\ndef heuristic('
    start_idx = specification.index(start_token)
    body_idx = specification.index(') -> int:\n', start_idx) + len(') -> int:\n')
    return specification[:body_idx] + new_body + "\n"


MULTI_TEMPLATE_SPECIFICATIONS: dict[str, str] = {
    "default": DEFAULT_SPECIFICATION,
    "front_biased": _replace_heuristic_body(
        DEFAULT_SPECIFICATION,
        """    \"\"\"Front-biased seed heuristic.\"\"\"
    seq_len = len(current_sequence)
    if seq_len == 0:
        return 0

    job_front = float(job_processing_times[0])
    job_back = float(job_processing_times[-1])
    current_tail = float(machine_completion_times[-1]) if len(machine_completion_times) > 0 else 0.0
    if current_job_release_time <= current_tail and job_front <= job_back:
        return 0
    if sequence_processing_times.size > 0:
        sequence_totals = np.sum(sequence_processing_times, axis=1)
        if float(np.sum(job_processing_times)) <= float(np.mean(sequence_totals)):
            return max(0, seq_len // 3)
    return seq_len // 2""",
    ),
    "back_biased": _replace_heuristic_body(
        DEFAULT_SPECIFICATION,
        """    \"\"\"Back-biased seed heuristic.\"\"\"
    seq_len = len(current_sequence)
    if seq_len == 0:
        return 0

    job_total = float(np.sum(job_processing_times))
    sequence_totals = np.sum(sequence_processing_times, axis=1) if sequence_processing_times.size > 0 else np.array([job_total], dtype=float)
    mean_total = float(np.mean(sequence_totals))
    mean_release = float(np.mean(sequence_release_times)) if len(sequence_release_times) > 0 else current_job_release_time
    if current_job_release_time >= mean_release or job_total >= mean_total:
        return seq_len
    return max(seq_len - 1, 0)""",
    ),
    "threshold_switch": _replace_heuristic_body(
        DEFAULT_SPECIFICATION,
        """    \"\"\"Threshold-switch seed heuristic.\"\"\"
    seq_len = len(current_sequence)
    if seq_len == 0:
        return 0

    job_total = float(np.sum(job_processing_times))
    current_tail = float(machine_completion_times[-1]) if len(machine_completion_times) > 0 else 0.0
    sequence_totals = np.sum(sequence_processing_times, axis=1) if sequence_processing_times.size > 0 else np.array([job_total], dtype=float)
    mean_total = float(np.mean(sequence_totals))
    release_anchor = float(np.mean(sequence_release_times)) if len(sequence_release_times) > 0 else current_job_release_time
    if current_job_release_time < release_anchor - 3.0:
        return 0
    if current_job_release_time > max(release_anchor + 3.0, current_tail):
        return seq_len
    if job_total <= 0.9 * mean_total:
        return max(0, seq_len // 3)
    if job_total >= 1.1 * mean_total:
        return min(seq_len, (2 * seq_len) // 3 + 1)
    return seq_len // 2""",
    ),
    "extreme_front": _replace_heuristic_body(
        DEFAULT_SPECIFICATION,
        """    \"\"\"Aggressively front-load early and light jobs.\"\"\"
    seq_len = len(current_sequence)
    if seq_len == 0:
        return 0

    job_total = float(np.sum(job_processing_times))
    job_front = float(job_processing_times[0])
    current_tail = float(machine_completion_times[-1]) if len(machine_completion_times) > 0 else current_job_release_time
    mean_release = float(np.mean(sequence_release_times)) if len(sequence_release_times) > 0 else current_job_release_time
    release_slack = current_tail - current_job_release_time

    if current_job_release_time <= mean_release or release_slack > job_front:
        return 0
    if sequence_processing_times.size > 0:
        sequence_totals = np.sum(sequence_processing_times, axis=1)
        if job_total <= float(np.percentile(sequence_totals, 35)):
            return min(1, seq_len)
    return max(0, seq_len // 4)""",
    ),
    "extreme_back": _replace_heuristic_body(
        DEFAULT_SPECIFICATION,
        """    \"\"\"Aggressively postpone heavy or late jobs.\"\"\"
    seq_len = len(current_sequence)
    if seq_len == 0:
        return 0

    job_total = float(np.sum(job_processing_times))
    mean_release = float(np.mean(sequence_release_times)) if len(sequence_release_times) > 0 else current_job_release_time
    mean_total = float(np.mean(np.sum(sequence_processing_times, axis=1))) if sequence_processing_times.size > 0 else job_total
    if current_job_release_time >= mean_release or job_total >= mean_total:
        return seq_len
    return max(seq_len - 1, 0)""",
    ),
    "release_aware_front": _replace_heuristic_body(
        DEFAULT_SPECIFICATION,
        """    \"\"\"Prefer front insertion for sufficiently early arrivals.\"\"\"
    seq_len = len(current_sequence)
    if seq_len == 0:
        return 0

    mean_release = float(np.mean(sequence_release_times)) if len(sequence_release_times) > 0 else current_job_release_time
    early_threshold = mean_release - float(np.std(sequence_release_times)) if len(sequence_release_times) > 1 else mean_release
    if current_job_release_time <= early_threshold:
        return 0
    if sequence_processing_times.size > 0:
        sequence_totals = np.sum(sequence_processing_times, axis=1)
        if float(np.sum(job_processing_times)) <= float(np.median(sequence_totals)):
            return max(0, seq_len // 3)
    return seq_len // 2""",
    ),
    "load_balancing_middle": _replace_heuristic_body(
        DEFAULT_SPECIFICATION,
        """    \"\"\"Bias toward middle insertion to smooth machine load.\"\"\"
    seq_len = len(current_sequence)
    if seq_len == 0:
        return 0

    current_tail = float(machine_completion_times[-1]) if len(machine_completion_times) > 0 else 0.0
    current_head = float(machine_completion_times[0]) if len(machine_completion_times) > 0 else 0.0
    imbalance = current_tail - current_head
    mid = seq_len // 2
    if imbalance > float(np.sum(job_processing_times)):
        return mid
    if current_job_release_time > current_tail:
        return seq_len
    return min(seq_len, mid + (seq_len % 2))""",
    ),
}


@dataclass
class SearchArtifacts:
    template: Any
    function_to_evolve: str
    best_program: Any
    best_score: float
    top_k_programs: list[Any]
    top_k_scores: list[float]
    search_diagnostics: dict[str, Any]
    template_name: str


class LLMAPI(sampler.LLM):
    """OpenAI-compatible LLM wrapper for FunSearch."""

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        api_key = HARDCODED_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("FUNSEARCH_API_KEY")
        base_url = HARDCODED_BASE_URL or os.getenv("OPENAI_BASE_URL") or os.getenv("FUNSEARCH_BASE_URL")
        model_name = HARDCODED_MODEL_NAME or os.getenv("FUNSEARCH_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        if not api_key:
            raise ValueError(
                "Missing API key. Please fill HARDCODED_API_KEY in the script or set OPENAI_API_KEY / FUNSEARCH_API_KEY."
            )

        client_kwargs = {"api_key": api_key, "timeout": 180}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._model_name = model_name
        self._raw_output_dir = PROJECT_DIR / "logs" / "online_flowshop_funsearch" / "llm_raw"
        self._raw_output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_counter = 0
        self._max_retries = MAX_LLM_RETRIES
        self._retry_sleep_seconds = 2.0
        self._additional_prompt = (
            "You are evolving a Python heuristic for ONLINE permutation flow-shop scheduling with release times. "
            "Jobs arrive one by one in a seeded random arrival order, and release times are generated from random inter-arrival gaps. "
            "At each step, the heuristic must choose the insertion index of the newly arrived job in the current partial sequence. "
            "The score is the average relative improvement over the initial heuristic baseline, so your goal is to produce a robust rule that slightly but consistently improves final makespan. "
            "Important dimension rules: current_sequence, sequence_processing_times, and sequence_release_times are indexed by sequence position; "
            "job_processing_times and machine_completion_times are indexed by machine. Never use machine indices to access sequence arrays, and never use sequence positions to access machine arrays. "
            "Prefer simple robust insertion rules over complex simulations. Use only the provided numeric inputs. Avoid constructing large temporary arrays or complicated pseudo-schedule simulations. "
            "Very important: produce behaviorally distinct heuristics rather than tiny variants of the default front/back/middle rule. "
            "Actively choose ONE clear strategy family and commit to it, for example: strongly front-biased insertion, strongly back-biased insertion, middle-stabilizing insertion, release-sensitive insertion, bottleneck-load-sensitive insertion, front-heavy-job prioritization, back-heavy-job postponement, or threshold-based regime switching. "
            "Use explicit thresholds, scoring formulas, or regime conditions so that the policy's insertion behavior is meaningfully different from common append/front/middle defaults. "
            "Avoid copying the baseline logic structure. Do not just restate: late->back, early->front, else->middle. Instead, create a distinct decision rule with a recognizable insertion style. "
            "When using loops, evaluate candidate insertion positions with a lightweight custom surrogate score based only on provided arrays; do not use invalid indexing tricks, fake placeholder jobs, or pseudo-simulations that mix sequence indices with machine indices. "
            "Always return exactly one integer insertion index in [0, len(current_sequence)]. "
            "Return only the BODY of function heuristic. No markdown, no code fences, no explanation."
        )

    def draw_samples(self, prompt: str) -> Collection[str]:
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        prompt = f"{content}\n{self._additional_prompt}"
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=1200,
                )
                raw_output = response.choices[0].message.content or ""
                cleaned_output = self._extract_heuristic_body(raw_output)
                self._save_sample(raw_output, cleaned_output)
                return cleaned_output
            except Exception as exc:
                last_error = exc
                print(
                    f"[WARN] LLM sampling failed (attempt {attempt}/{self._max_retries}): {exc}"
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_sleep_seconds)

        raise RuntimeError(
            "LLM sampling failed after "
            f"{self._max_retries} attempts. Check HARDCODED_API_KEY / HARDCODED_BASE_URL / HARDCODED_MODEL_NAME. "
            f"Last error: {last_error}"
        )

    def _save_sample(self, raw_output: str, cleaned_output: str) -> None:
        path = self._raw_output_dir / f"sample_{self._sample_counter:05d}.txt"
        self._sample_counter += 1
        path.write_text(
            "=== RAW OUTPUT ===\n"
            f"{raw_output}\n\n"
            "=== CLEANED HEURISTIC BODY ===\n"
            f"{cleaned_output}\n",
            encoding="utf-8",
        )

    @staticmethod
    def _extract_heuristic_body(text: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)

        lines = text.splitlines()
        def_index = None
        for idx, line in enumerate(lines):
            if re.match(r"^\s*def\s+heuristic\s*\(", line):
                def_index = idx
                break

        if def_index is not None:
            body_lines: list[str] = []
            for line in lines[def_index + 1:]:
                if not line.strip():
                    body_lines.append(line)
                    continue
                if line.startswith((" ", "\t")):
                    body_lines.append(line)
                    continue
                break
            text = "\n".join(body_lines)
        else:
            text = "\n".join(lines)

        text = text.strip("\n")
        if not text.strip():
            return "    return len(current_sequence)\n"

        stripped_lines = []
        for line in text.splitlines():
            if re.match(r"^\s*@", line):
                continue
            if re.match(r"^\s*def\s+", line):
                continue
            stripped_lines.append(line)
        text = "\n".join(stripped_lines).strip("\n")

        non_empty = [line for line in text.splitlines() if line.strip()]
        already_indented = bool(non_empty) and all(line.startswith((" ", "\t")) for line in non_empty)
        if not already_indented:
            text = "\n".join(f"    {line}" if line.strip() else "" for line in text.splitlines())

        body = text.rstrip()
        fallback_body = [
            "    seq_len = len(current_sequence)",
            "    try:",
        ]
        if body:
            fallback_body.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
        else:
            fallback_body.append("        return seq_len")
        fallback_body.extend([
            "    except Exception:",
            "        return seq_len",
        ])
        return "\n".join(fallback_body).rstrip() + "\n"


class Sandbox(evaluator.Sandbox):
    """Lightweight sandbox for executing generated code."""

    def run(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        inputs: Any,
        test_input: str,
        timeout_seconds: int,
    ) -> tuple[Any, bool]:
        del function_to_evolve, timeout_seconds
        instance = inputs[test_input]
        try:
            namespace: dict[str, Any] = {}
            exec(program, namespace)
            result = namespace[function_to_run](instance)
            if not isinstance(result, (int, float, np.floating, np.integer)):
                return None, False
            return float(result), True
        except Exception as exc:
            print(f"[WARN] Heuristic evaluation failed on {test_input}: {type(exc).__name__}: {exc}")
            return None, False


def load_taillard_instance(file_path: str) -> dict[str, Any]:
    """Loads one Taillard flow-shop instance into (jobs, machines) format."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError(f"Empty dataset file: {file_path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid Taillard header in file: {file_path}")

    num_jobs = int(header[0])
    num_machines = int(header[1])
    values: list[int] = []
    for line in lines[1:]:
        values.extend(int(token) for token in line.split())

    expected_values = num_jobs * num_machines
    if len(values) < expected_values:
        raise ValueError(
            f"Invalid Taillard matrix in {file_path}: expected {expected_values} values, got {len(values)}."
        )

    matrix = np.asarray(values[:expected_values], dtype=float).reshape(num_machines, num_jobs).T
    return {
        "file_name": Path(file_path).name,
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "processing_times": matrix,
    }


def compute_makespan_for_sequence(sequence: np.ndarray, processing_times: np.ndarray, release_times: np.ndarray) -> float:
    """Computes makespan for a fixed sequence under release times."""
    num_machines = processing_times.shape[1]
    completion = np.zeros(num_machines, dtype=float)
    for job_id in sequence:
        job_id = int(job_id)
        completion[0] = max(completion[0], float(release_times[job_id])) + float(processing_times[job_id, 0])
        for machine_id in range(1, num_machines):
            completion[machine_id] = max(completion[machine_id], completion[machine_id - 1]) + float(processing_times[job_id, machine_id])
    return float(completion[-1]) if len(sequence) > 0 else 0.0


def online_fcfs_baseline(processing_times: np.ndarray, release_times: np.ndarray, arrival_order: np.ndarray) -> tuple[np.ndarray, float]:
    """Baseline online policy: append jobs in arrival order."""
    sequence = np.asarray(arrival_order, dtype=int).copy()
    makespan = compute_makespan_for_sequence(sequence, processing_times, release_times)
    return sequence, makespan


def compute_initial_heuristic_baseline(
    runtime_namespace: dict[str, Any],
    processing_times: np.ndarray,
    release_times: np.ndarray,
    arrival_order: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Baseline online policy: run the initial heuristic from the specification."""
    simulate_fn = runtime_namespace["simulate_online_schedule"]
    sequence, makespan = simulate_fn(
        np.asarray(processing_times, dtype=float),
        np.asarray(release_times, dtype=float),
        np.asarray(arrival_order, dtype=int),
    )
    return np.asarray(sequence, dtype=int), float(makespan)


def attach_online_arrivals(instances: dict[str, dict[str, Any]], seed: int, min_gap: int, max_gap: int) -> None:
    """Adds reproducible random job arrival order and inter-arrival times to each instance."""
    if max_gap < min_gap:
        raise ValueError("arrival-gap-max must be >= arrival-gap-min")

    for file_name, instance in tqdm(
        instances.items(),
        total=len(instances),
        desc="Generating online arrivals",
        leave=False,
    ):
        num_jobs = int(instance["num_jobs"])
        seed_src = f"{seed}:{file_name}"
        derived_seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(derived_seed)

        arrival_order = rng.permutation(num_jobs).astype(int)
        gaps = rng.integers(min_gap, max_gap + 1, size=num_jobs, dtype=int)
        arrival_times = np.cumsum(gaps, dtype=float)
        if num_jobs > 0:
            arrival_times -= arrival_times[0]

        release_times = np.zeros(num_jobs, dtype=float)
        if num_jobs > 0:
            release_times[arrival_order] = arrival_times

        instance["release_times"] = release_times.astype(float)
        instance["arrival_order"] = arrival_order.astype(int)


def attach_baseline_makespans(instances: dict[str, dict[str, Any]], runtime_namespace: dict[str, Any]) -> None:
    """Adds baseline makespans used to normalize the score."""
    for instance in tqdm(
        instances.values(),
        total=len(instances),
        desc="Computing baselines",
        leave=False,
    ):
        _, baseline_makespan = compute_initial_heuristic_baseline(
            runtime_namespace=runtime_namespace,
            processing_times=np.asarray(instance["processing_times"], dtype=float),
            release_times=np.asarray(instance["release_times"], dtype=float),
            arrival_order=np.asarray(instance["arrival_order"], dtype=int),
        )
        instance["baseline_makespan"] = float(max(baseline_makespan, 1e-9))


def load_instances_from_directory(directory: str) -> dict[str, dict[str, Any]]:
    """Loads all .txt instances under a directory."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    instances = {}
    for file_path in sorted(path.glob("*.txt")):
        instances[file_path.name] = load_taillard_instance(str(file_path))
    if not instances:
        raise FileNotFoundError(f"No .txt dataset files found in: {directory}")
    return instances


def limit_instances(instances: dict[str, dict[str, Any]], limit: int) -> dict[str, dict[str, Any]]:
    """Keeps only the first `limit` instances when limit > 0."""
    if limit <= 0:
        return instances
    return dict(list(instances.items())[:limit])


def run_funsearch_search(
    specification: str,
    inputs: dict[str, dict[str, Any]],
    funsearch_config: config.Config,
    class_config: config.ClassConfig,
    max_sample_nums: int,
    log_dir: str,
    expert_top_k: int = 5,
    fingerprint_num_probes: int = 4,
    fingerprint_duplicate_threshold: int = 2,
    committee_score_weight: float = 0.7,
    committee_diversity_weight: float = 0.3,
    verbosity: str = "normal",
) -> SearchArtifacts:
    """Runs FunSearch, then extracts a diverse expert committee."""
    function_to_evolve, function_to_run = funsearch._extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = programs_database.ProgramsDatabase(
        funsearch_config.programs_database,
        template,
        function_to_evolve,
    )
    profiler = profile.Profiler(log_dir)

    evaluators = [
        evaluator.Evaluator(
            database=database,
            template=template,
            function_to_evolve=function_to_evolve,
            function_to_run=function_to_run,
            inputs=inputs,
            timeout_seconds=funsearch_config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
        )
        for _ in range(funsearch_config.num_evaluators)
    ]

    initial_body = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial_body, island_id=None, version_generated=None, profiler=profiler)

    sampler.Sampler._global_samples_nums = 1
    samplers = [
        sampler.Sampler(
            database=database,
            evaluators=evaluators,
            samples_per_prompt=funsearch_config.samples_per_prompt,
            max_sample_nums=max_sample_nums,
            llm_class=class_config.llm_class,
        )
        for _ in range(funsearch_config.num_samplers)
    ]
    for sampler_instance in tqdm(samplers, total=len(samplers), desc="FunSearch sampling"):
        sampler_instance.sample(profiler=profiler)

    fingerprinter = BehavioralFingerprinter(
        num_jobs=6,
        num_machines=4,
        num_probes=fingerprint_num_probes,
        duplicate_distance_threshold=fingerprint_duplicate_threshold,
    )
    valid_candidates: list[dict[str, Any]] = []
    exact_duplicate_experts = 0
    fingerprint_start_time = time.perf_counter()

    if verbosity != "quiet":
        print("\n" + "-" * 60)
        print("[INFO] Extracting diverse expert committee with behavioral fingerprints...")
    fingerprint_group_counts: dict[tuple[int, ...], int] = {}
    fingerprint_group_islands: dict[tuple[int, ...], list[int]] = {}
    fingerprint_group_scores: dict[tuple[int, ...], list[float]] = {}
    for island_id in range(len(database._best_score_per_island)):
        prog = database._best_program_per_island[island_id]
        score = database._best_score_per_island[island_id]
        if prog is None or not np.isfinite(score):
            continue

        runtime_template = code_manipulation.text_to_program(str(template))
        runtime_template.get_function(function_to_evolve).body = prog.body
        namespace: dict[str, Any] = {}
        try:
            exec(str(runtime_template), namespace)
            heuristic_func = namespace[function_to_evolve]
            fingerprint = fingerprinter.get_fingerprint(heuristic_func)
            if fingerprint is None:
                if verbosity == "debug":
                    print(f"  [DROP] island={island_id:2d} | reason=probe execution failure")
                continue
            is_exact_duplicate = any(fingerprint == item["fingerprint"] for item in valid_candidates)
            fingerprint_group_counts[fingerprint] = fingerprint_group_counts.get(fingerprint, 0) + 1
            fingerprint_group_islands.setdefault(fingerprint, []).append(island_id)
            fingerprint_group_scores.setdefault(fingerprint, []).append(float(score))
            if is_exact_duplicate:
                exact_duplicate_experts += 1
                if verbosity == "debug":
                    print(f"  [CAND] island={island_id:2d} | score={score:8.5f} | fp_len={len(fingerprint)} | note=exact fingerprint duplicate kept for committee selection")
            else:
                if verbosity == "debug":
                    print(f"  [CAND] island={island_id:2d} | score={score:8.5f} | fp_len={len(fingerprint)}")
            valid_candidates.append(
                {
                    "score": float(score),
                    "program": prog,
                    "fingerprint": fingerprint,
                    "island_id": island_id,
                    "is_exact_duplicate": is_exact_duplicate,
                }
            )
        except Exception as exc:
            if verbosity == "debug":
                print(f"  [DROP] island={island_id:2d} | reason=compile failure: {type(exc).__name__}")

    committee_selection_start_time = time.perf_counter()
    representative_committee = fingerprinter.select_representative_committee(
        valid_candidates,
        top_k=expert_top_k,
        score_weight=committee_score_weight,
        diversity_weight=committee_diversity_weight,
    )
    committee_selection_elapsed = time.perf_counter() - committee_selection_start_time
    fingerprinting_elapsed = time.perf_counter() - fingerprint_start_time
    top_k_programs = [item["program"] for item in representative_committee]
    top_k_scores = [float(item["score"]) for item in representative_committee]

    if top_k_programs:
        best_program = top_k_programs[0]
        best_score = top_k_scores[0]
    else:
        best_island_id = int(np.argmax(database._best_score_per_island))
        best_program = database._best_program_per_island[best_island_id]
        best_score = float(database._best_score_per_island[best_island_id])

    candidate_experts = int(sum(1 for score in database._best_score_per_island if np.isfinite(score)))
    exact_duplicates_retained = int(sum(1 for item in valid_candidates if item.get("is_exact_duplicate", False)))
    unique_fingerprints_pre_committee = len({item["fingerprint"] for item in valid_candidates})
    raw_to_unique_redundancy_ratio = float(1.0 - (unique_fingerprints_pre_committee / max(candidate_experts, 1))) if candidate_experts > 0 else 0.0
    committee_diversity_scores = [float(item.get("diversity_score", 0.0)) for item in representative_committee[1:]]
    mean_committee_diversity = float(np.mean(committee_diversity_scores)) if committee_diversity_scores else 0.0
    committee_unique_fingerprints = len({item["fingerprint"] for item in representative_committee}) if representative_committee else 0
    candidate_to_committee_compression = float(candidate_experts / max(len(top_k_programs), 1)) if candidate_experts > 0 else 0.0
    unique_to_committee_compression = float(unique_fingerprints_pre_committee / max(committee_unique_fingerprints, 1)) if unique_fingerprints_pre_committee > 0 else 0.0
    committee_unique_utilization_ratio = float(committee_unique_fingerprints / max(len(top_k_programs), 1)) if top_k_programs else 0.0
    fingerprint_group_summary = sorted(
        [
            {
                "count": count,
                "islands": fingerprint_group_islands[fingerprint],
                "scores": fingerprint_group_scores[fingerprint],
            }
            for fingerprint, count in fingerprint_group_counts.items()
        ],
        key=lambda item: (-int(item["count"]), int(item["islands"][0]) if item["islands"] else 0),
    )
    search_diagnostics = {
        "candidate_experts": candidate_experts,
        "kept_experts": len(top_k_programs),
        "exact_duplicate_experts": exact_duplicate_experts,
        "exact_duplicates_retained": exact_duplicates_retained,
        "unique_fingerprints_pre_committee": unique_fingerprints_pre_committee,
        "committee_unique_fingerprints": committee_unique_fingerprints,
        "candidate_to_committee_compression": candidate_to_committee_compression,
        "unique_to_committee_compression": unique_to_committee_compression,
        "raw_to_unique_redundancy_ratio": raw_to_unique_redundancy_ratio,
        "committee_unique_utilization_ratio": committee_unique_utilization_ratio,
        "fingerprint_group_count": len(fingerprint_group_summary),
        "fingerprint_group_summary": fingerprint_group_summary,
        "fingerprinting_elapsed_seconds": float(fingerprinting_elapsed),
        "committee_selection_elapsed_seconds": float(committee_selection_elapsed),
        "mean_committee_diversity": mean_committee_diversity,
    }

    if verbosity != "quiet":
        print("=" * 60)
        print(f"[INFO] Diverse expert committee kept: {len(top_k_programs)} / {expert_top_k}")
        print(f"[INFO] Exact duplicate experts observed: {exact_duplicate_experts}")
        print(f"[INFO] Exact duplicate experts retained for committee selection: {exact_duplicates_retained}")
        print(f"[INFO] Unique fingerprints before committee selection: {unique_fingerprints_pre_committee}")
        print(f"[INFO] Unique fingerprints inside committee: {committee_unique_fingerprints}")
        print(f"[INFO] Candidate-to-committee compression ratio: {candidate_to_committee_compression:.2f}x")
        print(f"[INFO] Unique-to-committee compression ratio: {unique_to_committee_compression:.2f}x")
        print(f"[INFO] Raw-to-unique redundancy ratio: {raw_to_unique_redundancy_ratio:.4f}")
        print(f"[INFO] Committee unique utilization ratio: {committee_unique_utilization_ratio:.4f}")
        print(f"[INFO] Fingerprinting group count: {len(fingerprint_group_summary)}")
        if verbosity == "debug":
            for group_idx, group in enumerate(fingerprint_group_summary[:3], start=1):
                print(
                    f"[INFO] Fingerprint group {group_idx}: size={int(group['count'])} "
                    f"islands={group['islands']} scores={[round(float(value), 5) for value in group['scores']] }"
                )
        print(f"[INFO] Fingerprinting time: {fingerprinting_elapsed:.4f}s")
        print(f"[INFO] Committee selection time: {committee_selection_elapsed:.4f}s")
        print(f"[INFO] Mean committee diversity score: {mean_committee_diversity:.4f}")
        print("=" * 60)

    return SearchArtifacts(
        template=template,
        function_to_evolve=function_to_evolve,
        best_program=best_program,
        best_score=float(best_score),
        top_k_programs=top_k_programs,
        top_k_scores=top_k_scores,
        search_diagnostics=search_diagnostics,
        template_name="single_template_run",
    )


def build_runtime_namespace(template: Any, function_to_evolve: str, best_program: Any) -> dict[str, Any]:
    """Builds a runtime namespace containing the evolved heuristic and helpers."""
    runtime_template = code_manipulation.text_to_program(str(template))
    runtime_template.get_function(function_to_evolve).body = best_program.body
    namespace: dict[str, Any] = {}
    exec(str(runtime_template), namespace)
    return namespace


def simulate_single_best_with_trace(namespace: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    """Runs the single-best heuristic and records insertion decisions."""
    heuristic_fn = namespace["heuristic"]
    compute_completion_vector_fn = namespace["compute_completion_vector"]
    compute_flowshop_makespan_fn = namespace["compute_flowshop_makespan"]

    processing_times = np.asarray(instance["processing_times"], dtype=float)
    release_times = np.asarray(instance["release_times"], dtype=float)
    arrival_order = np.asarray(instance["arrival_order"], dtype=int)
    num_jobs, num_machines = processing_times.shape

    current_sequence = np.zeros(0, dtype=int)
    machine_completion_times = np.zeros(num_machines, dtype=float)
    insertion_trace: list[int] = []

    for step, job_id in enumerate(arrival_order):
        job_id = int(job_id)
        sequence_processing_times = (
            processing_times[current_sequence]
            if len(current_sequence) > 0 else np.zeros((0, num_machines), dtype=float)
        )
        sequence_release_times = (
            release_times[current_sequence]
            if len(current_sequence) > 0 else np.zeros(0, dtype=float)
        )

        try:
            insert_index = heuristic_fn(
                current_sequence=current_sequence.copy(),
                job_processing_times=processing_times[job_id].copy(),
                machine_completion_times=machine_completion_times.copy(),
                sequence_processing_times=sequence_processing_times.copy(),
                current_job_release_time=float(release_times[job_id]),
                sequence_release_times=sequence_release_times.copy(),
            )
        except TypeError:
            try:
                insert_index = heuristic_fn(
                    current_sequence.copy(),
                    processing_times[job_id].copy(),
                    machine_completion_times.copy(),
                    sequence_processing_times.copy(),
                    float(release_times[job_id]),
                    sequence_release_times.copy(),
                )
            except Exception:
                insert_index = len(current_sequence)
        except Exception:
            insert_index = len(current_sequence)

        if not isinstance(insert_index, (int, np.integer, float, np.floating)):
            insert_index = len(current_sequence)
        insert_index = int(insert_index)
        insert_index = max(0, min(insert_index, len(current_sequence)))
        insertion_trace.append(insert_index)

        new_sequence = np.empty(step + 1, dtype=int)
        if insert_index > 0:
            new_sequence[:insert_index] = current_sequence[:insert_index]
        new_sequence[insert_index] = job_id
        if insert_index < len(current_sequence):
            new_sequence[insert_index + 1:] = current_sequence[insert_index:]
        current_sequence = new_sequence
        machine_completion_times = compute_completion_vector_fn(current_sequence, processing_times, release_times)

    makespan = compute_flowshop_makespan_fn(current_sequence, processing_times, release_times)
    return {
        "sequence": current_sequence.tolist(),
        "makespan": float(makespan),
        "insertion_trace": insertion_trace,
    }


def simulate_fusion_with_trace(template: Any, function_to_evolve: str, fusion_module: DynamicHeuristicFusion, instance: dict[str, Any]) -> dict[str, Any]:
    """Runs the state-aware fusion committee and records dynamic diagnostics."""
    namespace: dict[str, Any] = {}
    exec(str(template), namespace)
    compute_completion_vector_fn = namespace["compute_completion_vector"]
    compute_flowshop_makespan_fn = namespace["compute_flowshop_makespan"]

    processing_times = np.asarray(instance["processing_times"], dtype=float)
    release_times = np.asarray(instance["release_times"], dtype=float)
    arrival_order = np.asarray(instance["arrival_order"], dtype=int)
    num_jobs, num_machines = processing_times.shape

    current_sequence = np.zeros(0, dtype=int)
    machine_completion_times = np.zeros(num_machines, dtype=float)
    insertion_trace: list[int] = []
    dynamic_weight_snapshots: list[np.ndarray] = []

    for step, job_id in enumerate(arrival_order):
        job_id = int(job_id)
        sequence_processing_times = (
            processing_times[current_sequence]
            if len(current_sequence) > 0 else np.zeros((0, num_machines), dtype=float)
        )
        sequence_release_times = (
            release_times[current_sequence]
            if len(current_sequence) > 0 else np.zeros(0, dtype=float)
        )

        insert_index = fusion_module.fused_heuristic(
            current_sequence=current_sequence.copy(),
            job_processing_times=processing_times[job_id].copy(),
            machine_completion_times=machine_completion_times.copy(),
            sequence_processing_times=sequence_processing_times.copy(),
            current_job_release_time=float(release_times[job_id]),
            sequence_release_times=sequence_release_times.copy(),
        )
        insert_index = int(max(0, min(insert_index, len(current_sequence))))
        insertion_trace.append(insert_index)
        if getattr(fusion_module, "last_dynamic_weights", None) is not None and len(fusion_module.last_dynamic_weights) > 0:
            dynamic_weight_snapshots.append(np.asarray(fusion_module.last_dynamic_weights, dtype=float))

        new_sequence = np.empty(step + 1, dtype=int)
        if insert_index > 0:
            new_sequence[:insert_index] = current_sequence[:insert_index]
        new_sequence[insert_index] = job_id
        if insert_index < len(current_sequence):
            new_sequence[insert_index + 1:] = current_sequence[insert_index:]
        current_sequence = new_sequence
        machine_completion_times = compute_completion_vector_fn(current_sequence, processing_times, release_times)

    makespan = compute_flowshop_makespan_fn(current_sequence, processing_times, release_times)
    return {
        "sequence": current_sequence.tolist(),
        "makespan": float(makespan),
        "insertion_trace": insertion_trace,
        "dynamic_weight_snapshots": dynamic_weight_snapshots,
    }


def evaluate_instance_with_best(namespace: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    """Runs the best heuristic on a single instance and returns full results."""
    single_result = simulate_single_best_with_trace(namespace, instance)
    baseline_makespan = float(instance["baseline_makespan"])
    improvement = (baseline_makespan - float(single_result["makespan"])) / max(baseline_makespan, 1e-9)
    return {
        "file_name": instance["file_name"],
        "baseline_makespan": baseline_makespan,
        "best_makespan": float(single_result["makespan"]),
        "relative_improvement": float(improvement),
        "arrival_order": np.asarray(instance["arrival_order"], dtype=int).tolist(),
        "sequence": single_result["sequence"],
        "insertion_trace": single_result["insertion_trace"],
    }


def write_best_heuristic(best_program: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(best_program) + "\n", encoding="utf-8")


def write_test_results_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_name",
                "baseline_makespan",
                "best_makespan",
                "relative_improvement",
                "arrival_order",
                "final_sequence",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_name": row["file_name"],
                    "baseline_makespan": f"{row['baseline_makespan']:.6f}",
                    "best_makespan": f"{row['best_makespan']:.6f}",
                    "relative_improvement": f"{row['relative_improvement']:.6f}",
                    "arrival_order": json.dumps(row["arrival_order"], ensure_ascii=False),
                    "final_sequence": json.dumps(row["sequence"], ensure_ascii=False),
                }
            )


def write_test_results_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FunSearch for online Taillard flow-shop scheduling")
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(PROJECT_DIR / "Taillard" / "Train"),
        help="Training dataset directory.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(PROJECT_DIR / "Taillard" / "Verification"),
        help="Test dataset directory.",
    )
    parser.add_argument("--train-limit", type=int, default=20, help="Use only the first N training instances when > 0.")
    parser.add_argument("--test-limit", type=int, default=2, help="Use only the first N test instances when > 0.")
    parser.add_argument("--online-seed", type=int, default=42, help="Base seed for random release times.")
    parser.add_argument("--arrival-gap-min", type=int, default=0, help="Minimum gap between consecutive arrivals.")
    parser.add_argument("--arrival-gap-max", type=int, default=20, help="Maximum gap between consecutive arrivals.")
    parser.add_argument("--max-sample-nums", type=int, default=96, help="Maximum FunSearch samples.")
    parser.add_argument("--samples-per-prompt", type=int, default=4, help="Samples generated per prompt.")
    parser.add_argument("--template-mode", type=str, default="multi", choices=["single", "multi"], help="Use one default initial template or multiple seed templates.")
    parser.add_argument("--template-family", type=str, default="expanded", choices=["standard", "expanded"], help="Use the original template family or an expanded high-diversity seed family.")
    parser.add_argument("--template-order-mode", type=str, default="fixed", choices=["fixed", "rotate"], help="Keep template order fixed or rotate the order across runs to perturb search exposure.")
    parser.add_argument("--dedup-mode", type=str, default="on", choices=["on", "off"], help="Enable or disable fingerprint-based expert deduplication before committee selection.")
    parser.add_argument("--fusion-mode", type=str, default="dynamic", choices=["static", "dynamic"], help="Use static or dynamic expert weighting inside the fusion module.")
    parser.add_argument("--expert-top-k", type=int, default=5, help="Maximum number of experts kept in the committee.")
    parser.add_argument("--fusion-temperature", type=float, default=0.35, help="Temperature for state-aware expert fusion.")
    parser.add_argument("--fusion-spread-scale", type=float, default=0.35, help="Spread scale for per-position fusion kernels.")
    parser.add_argument("--fusion-dynamic-mix", type=float, default=0.35, help="How strongly dynamic gating deviates from base expert weights; 0 keeps static weights, 1 uses fully dynamic gating.")
    parser.add_argument("--fusion-min-expert-weight", type=float, default=0.08, help="Minimum per-expert probability floor before renormalization in dynamic fusion.")
    parser.add_argument("--fingerprint-num-probes", type=int, default=6, help="Number of probe instances used for behavioral fingerprinting.")
    parser.add_argument("--fingerprint-duplicate-threshold", type=int, default=8, help="Maximum fingerprint distance treated as duplicate.")
    parser.add_argument("--committee-score-weight", type=float, default=0.8, help="Representative committee selection weight for expert quality.")
    parser.add_argument("--committee-diversity-weight", type=float, default=0.2, help="Representative committee selection weight for diversity.")
    parser.add_argument("--robustness-alpha", type=float, default=0.2, help="Weight of robust downside penalty in evaluation summary.")
    parser.add_argument("--verbosity", type=str, default="normal", choices=["quiet", "normal", "debug"], help="Terminal logging level for large-scale experiments.")
    parser.add_argument("--log-dir", type=str, default=str(PROJECT_DIR / "logs" / "online_flowshop_funsearch"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_instances = load_instances_from_directory(args.train_dir)
    test_instances = load_instances_from_directory(args.test_dir)
    train_instances = limit_instances(train_instances, args.train_limit)
    test_instances = limit_instances(test_instances, args.test_limit)

    if not train_instances:
        raise ValueError("Training set is empty.")
    if not test_instances:
        raise ValueError("Test set is empty.")

    attach_online_arrivals(train_instances, args.online_seed, args.arrival_gap_min, args.arrival_gap_max)
    attach_online_arrivals(test_instances, args.online_seed, args.arrival_gap_min, args.arrival_gap_max)

    initial_template = code_manipulation.text_to_program(DEFAULT_SPECIFICATION)
    initial_function_to_evolve, _ = funsearch._extract_function_names(DEFAULT_SPECIFICATION)
    initial_runtime_namespace = build_runtime_namespace(
        initial_template,
        initial_function_to_evolve,
        initial_template.get_function(initial_function_to_evolve),
    )
    attach_baseline_makespans(train_instances, initial_runtime_namespace)
    attach_baseline_makespans(test_instances, initial_runtime_namespace)

    print(f"[INFO] Training instances: {len(train_instances)}")
    print(f"[INFO] Test instances: {len(test_instances)}")
    print(f"[INFO] Online arrival seed: {args.online_seed}")
    print(f"[INFO] Random inter-arrival gaps: [{args.arrival_gap_min}, {args.arrival_gap_max}]")
    print("[INFO] Arrival order is a seeded random permutation; release times follow random inter-arrival gaps.")
    print("[INFO] Score = average relative improvement over the initial heuristic baseline.")

    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    funsearch_config = config.Config(samples_per_prompt=args.samples_per_prompt)

    if args.template_family == "standard":
        selected_template_specs = {
            key: value
            for key, value in MULTI_TEMPLATE_SPECIFICATIONS.items()
            if key in {"default", "front_biased", "back_biased", "threshold_switch"}
        }
    else:
        selected_template_specs = MULTI_TEMPLATE_SPECIFICATIONS

    template_items = [("default", DEFAULT_SPECIFICATION)] if args.template_mode == "single" else list(selected_template_specs.items())
    if args.template_order_mode == "rotate" and len(template_items) > 1:
        rotation = args.online_seed % len(template_items)
        template_items = template_items[rotation:] + template_items[:rotation]
    per_template_sample_budget = max(1, int(np.ceil(args.max_sample_nums / max(len(template_items), 1))))
    per_template_committee_k = max(args.expert_top_k, min(per_template_sample_budget, args.expert_top_k * 3))
    start_time = time.time()
    all_candidate_entries: list[dict[str, Any]] = []
    per_template_artifacts: list[SearchArtifacts] = []

    for template_name, specification in template_items:
        template_log_dir = str(Path(args.log_dir) / f"template_{template_name}")
        print(f"[INFO] Running FunSearch for template: {template_name}")
        template_artifacts = run_funsearch_search(
            specification=specification,
            inputs=train_instances,
            funsearch_config=funsearch_config,
            class_config=class_config,
            max_sample_nums=per_template_sample_budget,
            log_dir=template_log_dir,
            expert_top_k=per_template_committee_k,
            fingerprint_num_probes=args.fingerprint_num_probes,
            fingerprint_duplicate_threshold=args.fingerprint_duplicate_threshold,
            committee_score_weight=args.committee_score_weight,
            committee_diversity_weight=args.committee_diversity_weight,
            verbosity=args.verbosity,
        )
        template_artifacts.template_name = template_name
        per_template_artifacts.append(template_artifacts)
        for rank, (program, score) in enumerate(zip(template_artifacts.top_k_programs, template_artifacts.top_k_scores)):
            all_candidate_entries.append(
                {
                    "template_name": template_name,
                    "template": template_artifacts.template,
                    "function_to_evolve": template_artifacts.function_to_evolve,
                    "program": program,
                    "score": float(score),
                    "rank": rank,
                }
            )

    if not per_template_artifacts:
        raise RuntimeError("No template searches were completed.")

    elapsed = time.time() - start_time
    print(f"[INFO] FunSearch finished in {elapsed:.2f}s across {len(template_items)} template(s)")
    print(f"[INFO] Per-template candidate retention target: {per_template_committee_k}")

    fingerprinter = BehavioralFingerprinter(
        num_jobs=6,
        num_machines=4,
        num_probes=args.fingerprint_num_probes,
        duplicate_distance_threshold=args.fingerprint_duplicate_threshold,
    )
    merged_candidates: list[dict[str, Any]] = []
    merged_raw_candidates: list[dict[str, Any]] = []
    exact_duplicate_experts = 0
    dedup_enabled = args.dedup_mode == "on"
    merged_fingerprinting_start_time = time.perf_counter()
    if args.verbosity != "quiet":
        print("-" * 60)
        print(f"[INFO] Merging template-wise experts into one representative committee... | dedup_mode={args.dedup_mode}")
    for candidate in all_candidate_entries:
        expert_namespace = build_runtime_namespace(candidate["template"], candidate["function_to_evolve"], candidate["program"])
        heuristic_func = expert_namespace[candidate["function_to_evolve"]]
        fingerprint = fingerprinter.get_fingerprint(heuristic_func)
        if fingerprint is None:
            continue
        merged_entry = {
            "score": candidate["score"],
            "program": candidate["program"],
            "fingerprint": fingerprint,
            "template_name": candidate["template_name"],
            "template": candidate["template"],
            "function_to_evolve": candidate["function_to_evolve"],
        }
        merged_raw_candidates.append(merged_entry)
        if dedup_enabled and any(fingerprint == item["fingerprint"] for item in merged_candidates):
            exact_duplicate_experts += 1
            continue
        merged_candidates.append(merged_entry)

    merged_fingerprinting_elapsed = time.perf_counter() - merged_fingerprinting_start_time
    dedup_start_time = time.perf_counter()
    if dedup_enabled:
        representative_committee = []
        selected_fingerprints: list[tuple[int, ...]] = []
        if args.verbosity == "debug":
            print(f"[INFO] Greedy dedup diagnostics | threshold={args.fingerprint_duplicate_threshold}")
        for candidate in sorted(merged_candidates, key=lambda item: float(item["score"]), reverse=True):
            fingerprint = candidate["fingerprint"]
            if selected_fingerprints:
                min_distance = min(fingerprinter.distance(fingerprint, seen) for seen in selected_fingerprints)
            else:
                min_distance = -1
            is_duplicate = fingerprinter.is_duplicate(fingerprint, selected_fingerprints)
            if is_duplicate:
                if args.verbosity == "debug":
                    print(
                        f"  [SKIP] template={candidate['template_name']:<18} "
                        f"score={float(candidate['score']):8.5f} min_fp_distance={min_distance}"
                    )
                continue
            candidate["diversity_score"] = float(
                0.0 if not selected_fingerprints else fingerprinter.committee_diversity_score(fingerprint, selected_fingerprints)
            )
            if args.verbosity == "debug":
                print(
                    f"  [KEEP] template={candidate['template_name']:<18} "
                    f"score={float(candidate['score']):8.5f} min_fp_distance={min_distance}"
                )
            representative_committee.append(candidate)
            selected_fingerprints.append(fingerprint)
            if len(representative_committee) >= args.expert_top_k:
                break
        if len(representative_committee) < min(args.expert_top_k, len(merged_candidates)):
            if args.verbosity == "debug":
                print("[INFO] Dedup fallback: filling remaining committee slots by score.")
            for candidate in sorted(merged_candidates, key=lambda item: float(item["score"]), reverse=True):
                if candidate in representative_committee:
                    continue
                candidate.setdefault("diversity_score", 0.0)
                if args.verbosity == "debug":
                    print(
                        f"  [FILL] template={candidate['template_name']:<18} "
                        f"score={float(candidate['score']):8.5f}"
                    )
                representative_committee.append(candidate)
                if len(representative_committee) >= args.expert_top_k:
                    break
    else:
        representative_committee = sorted(merged_candidates, key=lambda item: float(item["score"]), reverse=True)[: args.expert_top_k]
        for item in representative_committee:
            item.setdefault("diversity_score", 0.0)
    dedup_elapsed = time.perf_counter() - dedup_start_time
    if representative_committee:
        best_candidate = representative_committee[0]
    else:
        fallback = max(all_candidate_entries, key=lambda item: item["score"])
        fallback_namespace = build_runtime_namespace(fallback["template"], fallback["function_to_evolve"], fallback["program"])
        fallback_fp = fingerprinter.get_fingerprint(fallback_namespace[fallback["function_to_evolve"]])
        best_candidate = {
            "score": fallback["score"],
            "program": fallback["program"],
            "fingerprint": fallback_fp,
            "template_name": fallback["template_name"],
            "template": fallback["template"],
            "function_to_evolve": fallback["function_to_evolve"],
            "diversity_score": 0.0,
        }
        representative_committee = [best_candidate]

    top_k_programs = [item["program"] for item in representative_committee]
    top_k_scores = [float(item["score"]) for item in representative_committee]
    mean_committee_diversity = float(np.mean([item.get("diversity_score", 0.0) for item in representative_committee[1:]])) if len(representative_committee) > 1 else 0.0
    merged_raw_candidate_count = len(merged_raw_candidates)
    merged_unique_candidate_count = len(merged_candidates)
    final_committee_count = len(representative_committee)
    merged_exact_dedup_ratio = float(exact_duplicate_experts / max(merged_raw_candidate_count, 1)) if dedup_enabled else 0.0
    merged_candidate_compression = float(merged_raw_candidate_count / max(final_committee_count, 1)) if merged_raw_candidate_count > 0 else 0.0
    merged_unique_compression = float(merged_unique_candidate_count / max(final_committee_count, 1)) if merged_unique_candidate_count > 0 else 0.0
    merged_raw_to_unique_redundancy_ratio = float(1.0 - (merged_unique_candidate_count / max(merged_raw_candidate_count, 1))) if merged_raw_candidate_count > 0 else 0.0
    merged_committee_unique_fingerprints = len({item["fingerprint"] for item in representative_committee}) if representative_committee else 0
    merged_committee_unique_utilization_ratio = float(merged_committee_unique_fingerprints / max(final_committee_count, 1)) if final_committee_count > 0 else 0.0
    artifacts = SearchArtifacts(
        template=best_candidate["template"],
        function_to_evolve=best_candidate["function_to_evolve"],
        best_program=best_candidate["program"],
        best_score=float(best_candidate["score"]),
        top_k_programs=top_k_programs,
        top_k_scores=top_k_scores,
        search_diagnostics={
            "candidate_experts": len(all_candidate_entries),
            "kept_experts": final_committee_count,
            "duplicate_experts": max(merged_raw_candidate_count - merged_unique_candidate_count, 0) if dedup_enabled else 0,
            "exact_duplicate_experts": exact_duplicate_experts if dedup_enabled else 0,
            "merged_raw_candidate_count": merged_raw_candidate_count,
            "merged_unique_candidate_count": merged_unique_candidate_count,
            "merged_exact_dedup_ratio": merged_exact_dedup_ratio,
            "merged_candidate_compression": merged_candidate_compression,
            "merged_unique_compression": merged_unique_compression,
            "merged_raw_to_unique_redundancy_ratio": merged_raw_to_unique_redundancy_ratio,
            "merged_committee_unique_fingerprints": merged_committee_unique_fingerprints,
            "merged_committee_unique_utilization_ratio": merged_committee_unique_utilization_ratio,
            "merged_fingerprinting_elapsed_seconds": float(merged_fingerprinting_elapsed),
            "merged_dedup_elapsed_seconds": float(dedup_elapsed),
            "mean_committee_diversity": mean_committee_diversity,
            "template_names": [item["template_name"] for item in representative_committee],
            "dedup_mode": args.dedup_mode,
            "fusion_mode": args.fusion_mode,
        },
        template_name=best_candidate["template_name"],
    )
    print(f"[INFO] Best training score: {artifacts.best_score:.6f} | source_template={artifacts.template_name}")
    print(f"[INFO] Ablation modes | dedup={args.dedup_mode} | fusion={args.fusion_mode}")
    print(f"[INFO] Merged committee kept: {artifacts.search_diagnostics['kept_experts']} / {args.expert_top_k}")
    print(f"[INFO] Merged raw candidates before dedup: {artifacts.search_diagnostics['merged_raw_candidate_count']}")
    print(f"[INFO] Merged unique candidates after exact dedup: {artifacts.search_diagnostics['merged_unique_candidate_count']}")
    print(f"[INFO] Merged exact duplicate experts filtered: {artifacts.search_diagnostics['exact_duplicate_experts']}")
    print(f"[INFO] Merged exact dedup ratio: {artifacts.search_diagnostics['merged_exact_dedup_ratio']:.4f}")
    print(f"[INFO] Merged candidate-to-committee compression: {artifacts.search_diagnostics['merged_candidate_compression']:.2f}x")
    print(f"[INFO] Merged unique-to-committee compression: {artifacts.search_diagnostics['merged_unique_compression']:.2f}x")
    print(f"[INFO] Merged raw-to-unique redundancy ratio: {artifacts.search_diagnostics['merged_raw_to_unique_redundancy_ratio']:.4f}")
    print(f"[INFO] Merged committee unique fingerprints: {artifacts.search_diagnostics['merged_committee_unique_fingerprints']}")
    print(f"[INFO] Merged committee unique utilization ratio: {artifacts.search_diagnostics['merged_committee_unique_utilization_ratio']:.4f}")
    print(f"[INFO] Merged fingerprinting time: {artifacts.search_diagnostics['merged_fingerprinting_elapsed_seconds']:.4f}s")
    print(f"[INFO] Merged dedup time: {artifacts.search_diagnostics['merged_dedup_elapsed_seconds']:.4f}s")
    print(f"[INFO] Merged mean committee diversity score: {artifacts.search_diagnostics['mean_committee_diversity']:.4f}")

    namespace = build_runtime_namespace(artifacts.template, artifacts.function_to_evolve, artifacts.best_program)
    expert_funcs: list[Any] = []
    for committee_item in representative_committee:
        expert_namespace = build_runtime_namespace(committee_item["template"], committee_item["function_to_evolve"], committee_item["program"])
        expert_funcs.append(expert_namespace[committee_item["function_to_evolve"]])

    fusion_module = DynamicHeuristicFusion(
        top_k=max(args.expert_top_k, 1),
        temperature=args.fusion_temperature,
        spread_scale=args.fusion_spread_scale,
        dynamic_mode=(args.fusion_mode == "dynamic"),
        dynamic_mix=args.fusion_dynamic_mix,
        min_expert_weight=args.fusion_min_expert_weight,
    )
    if expert_funcs:
        fusion_module.load_experts(expert_funcs, artifacts.top_k_scores)

    test_rows = []
    fusion_wins = 0
    single_wins = 0
    ties = 0
    same_insertion_count = 0
    total_insertion_steps = 0
    all_dynamic_weights: list[np.ndarray] = []

    for instance in tqdm(test_instances.values(), total=len(test_instances), desc="Evaluating test instances"):
        single_result = simulate_single_best_with_trace(namespace, instance)
        baseline_makespan = float(instance["baseline_makespan"])
        single_improvement = (baseline_makespan - float(single_result["makespan"])) / max(baseline_makespan, 1e-9)

        row = {
            "file_name": instance["file_name"],
            "baseline_makespan": baseline_makespan,
            "best_makespan": float(single_result["makespan"]),
            "relative_improvement": float(single_improvement),
            "arrival_order": np.asarray(instance["arrival_order"], dtype=int).tolist(),
            "sequence": single_result["sequence"],
            "insertion_trace": single_result["insertion_trace"],
        }

        if expert_funcs:
            fusion_result = simulate_fusion_with_trace(artifacts.template, artifacts.function_to_evolve, fusion_module, instance)
            fusion_makespan = float(fusion_result["makespan"])
            fusion_improvement = (baseline_makespan - fusion_makespan) / max(baseline_makespan, 1e-9)
            row["fusion_makespan"] = fusion_makespan
            row["fusion_relative_improvement"] = float(fusion_improvement)
            row["fusion_sequence"] = fusion_result["sequence"]
            row["fusion_insertion_trace"] = fusion_result["insertion_trace"]

            if fusion_makespan < row["best_makespan"]:
                fusion_wins += 1
            elif fusion_makespan > row["best_makespan"]:
                single_wins += 1
            else:
                ties += 1

            same_insertion_count += sum(
                1
                for single_pos, fusion_pos in zip(row["insertion_trace"], fusion_result["insertion_trace"])
                if single_pos == fusion_pos
            )
            total_insertion_steps += min(len(row["insertion_trace"]), len(fusion_result["insertion_trace"]))
            all_dynamic_weights.extend(fusion_result["dynamic_weight_snapshots"])

        test_rows.append(row)

    test_rows.sort(key=lambda row: row["best_makespan"])

    if test_rows and args.verbosity == "debug":
        best_row = test_rows[0]
        print("[RESULT] Best test instance under the evolved heuristic:")
        print(f"  file_name: {best_row['file_name']}")
        print(f"  min_makespan: {best_row['best_makespan']:.6f}")
        print(f"  baseline_makespan: {best_row['baseline_makespan']:.6f}")
        print(f"  relative_improvement: {best_row['relative_improvement']:.6f}")
        print(f"  arrival_order: {best_row['arrival_order']}")
        print(f"  final_sequence: {best_row['sequence']}")

    average_test_makespan = float(np.mean([row["best_makespan"] for row in test_rows]))
    average_test_improvement = float(np.mean([row["relative_improvement"] for row in test_rows]))
    print(f"[RESULT] Average single-best makespan: {average_test_makespan:.6f}")
    print(f"[RESULT] Average single-best relative improvement vs baseline: {average_test_improvement:.6f}")
    if args.verbosity == "debug":
        print("[RESULT] Per-instance single-best results:")
    for row in test_rows:
        print(
                f"  {row['file_name']}: single_best_makespan={row['best_makespan']:.6f}, "
                f"baseline={row['baseline_makespan']:.6f}, "
                f"single_best_improvement={row['relative_improvement']:.6f}"
            )

    robust_single_score = 0.0
    robust_fusion_score = 0.0
    if test_rows:
        single_improvements = np.asarray([row["relative_improvement"] for row in test_rows], dtype=float)
        worst_fraction = max(1, int(np.ceil(len(single_improvements) * max(args.robustness_alpha, 1e-6))))
        single_tail = np.sort(single_improvements)[:worst_fraction]
        robust_single_score = float(np.mean(single_improvements) - args.robustness_alpha * abs(np.mean(single_tail)))

    if expert_funcs:
        fusion_improvements = np.asarray([row["fusion_relative_improvement"] for row in test_rows], dtype=float)
        worst_fraction = max(1, int(np.ceil(len(fusion_improvements) * max(args.robustness_alpha, 1e-6))))
        fusion_tail = np.sort(fusion_improvements)[:worst_fraction]
        robust_fusion_score = float(np.mean(fusion_improvements) - args.robustness_alpha * abs(np.mean(fusion_tail)))

        avg_fusion_makespan = float(np.mean([row["fusion_makespan"] for row in test_rows]))
        avg_fusion_improvement = float(np.mean([row["fusion_relative_improvement"] for row in test_rows]))
        avg_fusion_gain_vs_single = float(np.mean([row["best_makespan"] - row["fusion_makespan"] for row in test_rows]))
        step_level_insertion_agreement_rate = same_insertion_count / max(total_insertion_steps, 1)
        if args.verbosity == "debug":
            print("[RESULT] Per-instance fusion results:")
            for row in test_rows:
                fusion_gain_vs_single = float(row["best_makespan"] - row["fusion_makespan"])
                print(
                    f"  {row['file_name']}: fusion_makespan={row['fusion_makespan']:.6f}, "
                    f"fusion_improvement_vs_baseline={row['fusion_relative_improvement']:.6f}, "
                    f"fusion_gain_vs_single={fusion_gain_vs_single:.6f}, "
                    f"single_best_makespan={row['best_makespan']:.6f}"
                )
        print("[RESULT] Fusion committee diagnostics:")
        print(f"  wins/ties/losses vs single best: {fusion_wins}/{ties}/{single_wins}")
        print(f"  average fusion makespan: {avg_fusion_makespan:.6f}")
        print(f"  average fusion relative improvement vs baseline: {avg_fusion_improvement:.6f}")
        print(f"  average fusion gain vs single best makespan: {avg_fusion_gain_vs_single:.6f}")
        print(f"  robust single-best score: {robust_single_score:.6f}")
        print(f"  robust fusion score: {robust_fusion_score:.6f}")
        print(f"  step-level insertion agreement rate: {step_level_insertion_agreement_rate:.6f}")
        print(f"  kept experts: {artifacts.search_diagnostics['kept_experts']}")
        print(f"  duplicate experts filtered: {artifacts.search_diagnostics['duplicate_experts']}")
        print(f"  exact duplicate experts filtered: {artifacts.search_diagnostics['exact_duplicate_experts']}")
        print(f"  merged raw candidates before dedup: {artifacts.search_diagnostics['merged_raw_candidate_count']}")
        print(f"  merged unique candidates after exact dedup: {artifacts.search_diagnostics['merged_unique_candidate_count']}")
        print(f"  merged exact dedup ratio: {artifacts.search_diagnostics['merged_exact_dedup_ratio']:.6f}")
        print(f"  merged candidate-to-committee compression: {artifacts.search_diagnostics['merged_candidate_compression']:.6f}")
        print(f"  merged unique-to-committee compression: {artifacts.search_diagnostics['merged_unique_compression']:.6f}")
        print(f"  merged raw-to-unique redundancy ratio: {artifacts.search_diagnostics['merged_raw_to_unique_redundancy_ratio']:.6f}")
        print(f"  merged committee unique fingerprints: {artifacts.search_diagnostics['merged_committee_unique_fingerprints']}")
        print(f"  merged committee unique utilization ratio: {artifacts.search_diagnostics['merged_committee_unique_utilization_ratio']:.6f}")
        print(f"  merged fingerprinting time: {artifacts.search_diagnostics['merged_fingerprinting_elapsed_seconds']:.6f}")
        print(f"  merged dedup time: {artifacts.search_diagnostics['merged_dedup_elapsed_seconds']:.6f}")
        print(f"  mean committee diversity score: {artifacts.search_diagnostics['mean_committee_diversity']:.6f}")
        print(f"  dedup mode: {artifacts.search_diagnostics['dedup_mode']}")
        print(f"  fusion mode: {artifacts.search_diagnostics['fusion_mode']}")
        if all_dynamic_weights:
            stacked_dynamic_weights = np.vstack(all_dynamic_weights)
            dynamic_weight_min = float(np.min(stacked_dynamic_weights))
            dynamic_weight_max = float(np.max(stacked_dynamic_weights))
            dynamic_weight_mean = np.round(np.mean(stacked_dynamic_weights, axis=0), 4).tolist()
            if len(all_dynamic_weights) >= 2:
                step_drifts = [
                    float(np.mean(np.abs(curr - prev)))
                    for prev, curr in zip(all_dynamic_weights[:-1], all_dynamic_weights[1:])
                ]
                dynamic_weight_drift_mean = float(np.mean(step_drifts))
                dynamic_weight_drift_max = float(np.max(step_drifts))
            else:
                dynamic_weight_drift_mean = 0.0
                dynamic_weight_drift_max = 0.0
            print(
                f"  dynamic weight range: min={dynamic_weight_min:.6f}, "
                f"max={dynamic_weight_max:.6f}, mean={dynamic_weight_mean}"
            )
            print(
                f"  dynamic weight drift: mean={dynamic_weight_drift_mean:.6f}, "
                f"max={dynamic_weight_drift_max:.6f}"
            )

    summary_payload = {
        "template_mode": args.template_mode,
        "template_family": args.template_family,
        "template_order_mode": args.template_order_mode,
        "dedup_mode": args.dedup_mode,
        "fusion_mode": args.fusion_mode,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "max_sample_nums": args.max_sample_nums,
        "samples_per_prompt": args.samples_per_prompt,
        "expert_top_k": args.expert_top_k,
        "fingerprint_num_probes": args.fingerprint_num_probes,
        "fingerprint_duplicate_threshold": args.fingerprint_duplicate_threshold,
        "committee_score_weight": args.committee_score_weight,
        "committee_diversity_weight": args.committee_diversity_weight,
        "fusion_temperature": args.fusion_temperature,
        "fusion_spread_scale": args.fusion_spread_scale,
        "fusion_dynamic_mix": args.fusion_dynamic_mix,
        "fusion_min_expert_weight": args.fusion_min_expert_weight,
        "robustness_alpha": args.robustness_alpha,
        "average_single_best_makespan": average_test_makespan,
        "average_single_best_relative_improvement": average_test_improvement,
        "robust_single_best_score": robust_single_score,
        "average_fusion_makespan": avg_fusion_makespan if expert_funcs else None,
        "average_fusion_relative_improvement": avg_fusion_improvement if expert_funcs else None,
        "average_fusion_gain_vs_single_best": avg_fusion_gain_vs_single if expert_funcs else None,
        "robust_fusion_score": robust_fusion_score if expert_funcs else None,
        "fusion_wins": fusion_wins if expert_funcs else None,
        "fusion_ties": ties if expert_funcs else None,
        "fusion_losses": single_wins if expert_funcs else None,
        "step_level_insertion_agreement_rate": step_level_insertion_agreement_rate if expert_funcs else None,
        "search_diagnostics": artifacts.search_diagnostics,
    }

    log_dir = Path(args.log_dir)
    write_best_heuristic(artifacts.best_program, log_dir / "best_heuristic.py")
    write_test_results_csv(test_rows, log_dir / "test_results.csv")
    write_test_results_json(test_rows, log_dir / "test_results.json")
    write_summary_json(summary_payload, log_dir / "summary.json")
    print(f"[INFO] Outputs saved under: {log_dir}")


if __name__ == "__main__":
    main()
