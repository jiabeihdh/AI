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


@dataclass
class SearchArtifacts:
    template: Any
    function_to_evolve: str
    best_program: Any
    best_score: float


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
) -> SearchArtifacts:
    """Runs FunSearch and returns the best evolved heuristic."""
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

    best_island_id = int(np.argmax(database._best_score_per_island))
    return SearchArtifacts(
        template=template,
        function_to_evolve=function_to_evolve,
        best_program=database._best_program_per_island[best_island_id],
        best_score=float(database._best_score_per_island[best_island_id]),
    )


def build_runtime_namespace(template: Any, function_to_evolve: str, best_program: Any) -> dict[str, Any]:
    """Builds a runtime namespace containing the evolved heuristic and helpers."""
    runtime_template = code_manipulation.text_to_program(str(template))
    runtime_template.get_function(function_to_evolve).body = best_program.body
    namespace: dict[str, Any] = {}
    exec(str(runtime_template), namespace)
    return namespace


def evaluate_instance_with_best(namespace: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    """Runs the best heuristic on a single instance and returns full results."""
    simulate_fn = namespace["simulate_online_schedule"]
    sequence, makespan = simulate_fn(
        np.asarray(instance["processing_times"], dtype=float),
        np.asarray(instance["release_times"], dtype=float),
        np.asarray(instance["arrival_order"], dtype=int),
    )
    baseline_makespan = float(instance["baseline_makespan"])
    improvement = (baseline_makespan - float(makespan)) / max(baseline_makespan, 1e-9)
    return {
        "file_name": instance["file_name"],
        "baseline_makespan": baseline_makespan,
        "best_makespan": float(makespan),
        "relative_improvement": float(improvement),
        "arrival_order": np.asarray(instance["arrival_order"], dtype=int).tolist(),
        "sequence": np.asarray(sequence, dtype=int).tolist(),
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
    parser.add_argument("--max-sample-nums", type=int, default=8, help="Maximum FunSearch samples.")
    parser.add_argument("--samples-per-prompt", type=int, default=4, help="Samples generated per prompt.")
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

    start_time = time.time()
    artifacts = run_funsearch_search(
        specification=DEFAULT_SPECIFICATION,
        inputs=train_instances,
        funsearch_config=funsearch_config,
        class_config=class_config,
        max_sample_nums=args.max_sample_nums,
        log_dir=args.log_dir,
    )
    elapsed = time.time() - start_time
    print(f"[INFO] FunSearch finished in {elapsed:.2f}s")
    print(f"[INFO] Best training score: {artifacts.best_score:.6f}")

    namespace = build_runtime_namespace(artifacts.template, artifacts.function_to_evolve, artifacts.best_program)
    test_rows = [
        evaluate_instance_with_best(namespace, instance)
        for instance in tqdm(test_instances.values(), total=len(test_instances), desc="Evaluating test instances")
    ]
    test_rows.sort(key=lambda row: row["best_makespan"])

    if test_rows:
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
    improvements = np.asarray([row["relative_improvement"] for row in test_rows], dtype=float)
    worst_fraction = max(1, int(np.ceil(len(improvements) * 0.2)))
    tail = np.sort(improvements)[:worst_fraction]
    robust_single_score = float(np.mean(improvements) - 0.2 * abs(np.mean(tail)))
    print(f"[RESULT] Average test makespan: {average_test_makespan:.6f}")
    print(f"[RESULT] Average test relative improvement: {average_test_improvement:.6f}")
    print(f"[RESULT] Robust single-best score: {robust_single_score:.6f}")
    print("[RESULT] Test set makespans:")
    for row in test_rows:
        print(
            f"  {row['file_name']}: makespan={row['best_makespan']:.6f}, "
            f"baseline={row['baseline_makespan']:.6f}, improvement={row['relative_improvement']:.6f}"
        )

    summary_payload = {
        "average_single_best_makespan": average_test_makespan,
        "average_single_best_relative_improvement": average_test_improvement,
        "robust_single_best_score": robust_single_score,
        "average_fusion_makespan": None,
        "average_fusion_relative_improvement": None,
        "average_fusion_gain_vs_single_best": None,
        "robust_fusion_score": None,
        "fusion_wins": None,
        "fusion_ties": None,
        "fusion_losses": None,
        "step_level_insertion_agreement_rate": None,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "max_sample_nums": args.max_sample_nums,
        "samples_per_prompt": args.samples_per_prompt,
        "search_diagnostics": {
            "kept_experts": 1,
            "merged_raw_candidate_count": None,
            "merged_unique_candidate_count": None,
            "merged_exact_dedup_ratio": None,
            "merged_candidate_compression": None,
            "merged_unique_compression": None,
            "merged_raw_to_unique_redundancy_ratio": None,
            "merged_committee_unique_fingerprints": None,
            "merged_committee_unique_utilization_ratio": None,
            "merged_fingerprinting_elapsed_seconds": None,
            "merged_dedup_elapsed_seconds": None,
            "mean_committee_diversity": None,
            "dedup_mode": "baseline",
            "fusion_mode": "baseline",
        },
    }

    log_dir = Path(args.log_dir)
    write_best_heuristic(artifacts.best_program, log_dir / "best_heuristic.py")
    write_test_results_csv(test_rows, log_dir / "test_results.csv")
    write_test_results_json(test_rows, log_dir / "test_results.json")
    write_summary_json(summary_payload, log_dir / "summary.json")
    print(f"[INFO] Outputs saved under: {log_dir}")


if __name__ == "__main__":
    main()
