import inspect
from typing import Callable, List, Optional, Union
import numpy as np

class DynamicHeuristicFusion:
    def __init__(self, top_k: int = 5, temperature: float = 0.35, spread_scale: float = 0.35, dynamic_mode: bool = True, dynamic_mix: float = 0.35, min_expert_weight: float = 0.08):
        self.top_k = top_k
        self.temperature = max(float(temperature), 1e-6)
        self.spread_scale = max(float(spread_scale), 1e-6)
        self.dynamic_mode = bool(dynamic_mode)
        self.dynamic_mix = min(max(float(dynamic_mix), 0.0), 1.0)
        self.min_expert_weight = min(max(float(min_expert_weight), 0.0), 1.0)
        self.heuristics: List[Callable] = []
        self.base_weights = np.array([], dtype=float)
        self.last_dynamic_weights = np.array([], dtype=float)

    def load_experts(self, heuristic_funcs: List[Callable], historical_scores: List[float]):
        self.heuristics = heuristic_funcs[: self.top_k]
        scores = np.asarray(historical_scores[: len(self.heuristics)], dtype=float)
        if len(scores) == 0:
            self.base_weights = np.array([], dtype=float)
        else:
            scores = scores - np.max(scores)
            weights = np.exp(scores / self.temperature)
            self.base_weights = weights / max(np.sum(weights), 1e-9)
        self.last_dynamic_weights = self.base_weights.copy()

    def _safe_call(self, func: Callable, args: list[Union[np.ndarray, float]]) -> Optional[int]:
        try:
            sig = inspect.signature(func)
            out = func(*args[: len(sig.parameters)])
            if isinstance(out, (int, np.integer)):
                return int(out)
        except Exception:
            return None
        return None

    def fused_heuristic(self, current_sequence, job_processing_times, machine_completion_times, sequence_processing_times, current_job_release_time, sequence_release_times) -> int:
        if not self.heuristics:
            return len(current_sequence)
        args = [current_sequence, job_processing_times, machine_completion_times, sequence_processing_times, current_job_release_time, sequence_release_times]
        votes = []
        for func in self.heuristics:
            pos = self._safe_call(func, args)
            if pos is not None:
                votes.append(max(0, min(len(current_sequence), pos)))
        if not votes:
            return len(current_sequence)
        if len(votes) == 1:
            return int(votes[0])
        return int(round(float(np.mean(votes))))
