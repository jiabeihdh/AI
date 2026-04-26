import inspect
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

ProbeFingerprint = Tuple[int, ...]

class BehavioralFingerprinter:
    def __init__(self, num_jobs: int = 5, num_machines: int = 3, seed: int = 42, num_probes: int = 4, duplicate_distance_threshold: int = 2):
        self.seed = seed
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_probes = max(int(num_probes), 1)
        self.duplicate_distance_threshold = max(int(duplicate_distance_threshold), 0)
        self.probes = self._build_probe_suite()

    def _build_probe_suite(self) -> List[dict]:
        probes = []
        for probe_idx in range(self.num_probes):
            rng = np.random.default_rng(self.seed + 97 * probe_idx)
            processing_times = rng.integers(1, 50, size=(self.num_jobs + probe_idx % 3, self.num_machines + probe_idx % 2)).astype(float)
            release_times = np.cumsum(rng.integers(0, 6, size=processing_times.shape[0]), dtype=float)
            release_times -= release_times[0]
            arrival_order = np.arange(processing_times.shape[0], dtype=int)
            probes.append({"processing_times": processing_times, "release_times": release_times, "arrival_order": arrival_order})
        return probes

    def _call_heuristic(self, heuristic_func: Callable, args: List[Union[np.ndarray, float]]) -> Optional[int]:
        try:
            sig = inspect.signature(heuristic_func)
            pos = heuristic_func(*args[: len(sig.parameters)])
            if isinstance(pos, (int, np.integer)):
                return int(pos)
        except Exception:
            return None
        return None

    def get_fingerprint(self, heuristic_func: Callable) -> Optional[Tuple[int, ...]]:
        values: list[int] = []
        for probe in self.probes:
            current_sequence: list[int] = []
            machine_completion_times = np.zeros(probe["processing_times"].shape[1], dtype=float)
            for job_id in probe["arrival_order"]:
                seq_proc = probe["processing_times"][np.array(current_sequence, dtype=int)] if current_sequence else np.zeros((0, probe["processing_times"].shape[1]), dtype=float)
                seq_release = np.array([probe["release_times"][j] for j in current_sequence], dtype=float) if current_sequence else np.zeros((0,), dtype=float)
                args = [np.array(current_sequence, dtype=int), probe["processing_times"][job_id], machine_completion_times, seq_proc, float(probe["release_times"][job_id]), seq_release]
                pos = self._call_heuristic(heuristic_func, args)
                if pos is None:
                    return None
                pos = max(0, min(len(current_sequence), pos))
                values.append(pos)
                current_sequence.insert(pos, int(job_id))
        return tuple(values)

    def distance(self, fingerprint_a: Tuple[int, ...], fingerprint_b: Tuple[int, ...]) -> int:
        max_len = max(len(fingerprint_a), len(fingerprint_b))
        return sum((fingerprint_a[i] if i < len(fingerprint_a) else None) != (fingerprint_b[i] if i < len(fingerprint_b) else None) for i in range(max_len))

    def is_duplicate(self, fingerprint: Tuple[int, ...], seen_fingerprints: List[Tuple[int, ...]]) -> bool:
        return any(self.distance(fingerprint, seen) <= self.duplicate_distance_threshold for seen in seen_fingerprints)

    def committee_diversity_score(self, fingerprint: Tuple[int, ...], committee_fingerprints: List[Tuple[int, ...]]) -> float:
        if not committee_fingerprints:
            return 0.0
        return float(np.mean([self.distance(fingerprint, seen) for seen in committee_fingerprints]))

    def select_representative_committee(self, candidates: List[dict], top_k: int = 5, score_weight: float = 0.7, diversity_weight: float = 0.3) -> List[dict]:
        if not candidates:
            return []
        sorted_candidates = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
        committee = [dict(sorted_candidates[0], diversity_score=0.0)]
        seen = [sorted_candidates[0]["fingerprint"]]
        while len(committee) < min(top_k, len(sorted_candidates)):
            best_candidate = None
            best_value = None
            for candidate in sorted_candidates:
                if any(candidate["program"] is member["program"] for member in committee):
                    continue
                diversity_score = self.committee_diversity_score(candidate["fingerprint"], seen)
                combined = score_weight * float(candidate["score"]) + diversity_weight * float(diversity_score)
                if best_value is None or combined > best_value:
                    best_value = combined
                    best_candidate = dict(candidate, diversity_score=float(diversity_score))
            if best_candidate is None:
                break
            committee.append(best_candidate)
            seen.append(best_candidate["fingerprint"])
        return committee
