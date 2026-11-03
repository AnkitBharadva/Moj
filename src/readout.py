"""
One-token letter readout for comment classification via Ollama.

Enhanced with:
1. Few-shot in-context exemplars for sarcasm, Hinglish, and political discourse
2. Descriptive option boundary definitions (eliminating personal insult vs hate confusion)
3. Cyclic Option-Order Debiasing (3-pass permutation averaging to eliminate position bias)
4. Dynamic target stance detection and arbitrary custom tasks
5. Clean key mapping: prompt uses rich descriptive options, output returns clean concise keys
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3.5-judge"
FALLBACK_MODEL = "comment-judge"
DEFAULT_URL = "http://localhost:11434/api/chat"
DEFAULT_TOP_K = 20
DEFAULT_NUM_CTX = 2048
DEFAULT_TIMEOUT = 60
DEFAULT_KEEP_ALIVE = -1
MAX_COMMENT_CHARS = 1200


@dataclass
class TaskConfig:
    """Configuration for a classification task."""
    name: str
    question: str
    options: List[str]  # Rich prompt descriptions (e.g. "positive (sincere praise...)")
    option_keys: List[str] = field(default_factory=list)  # Clean return keys (e.g. "positive")
    examples: List[str] = field(default_factory=list)  # Few-shot exemplars
    default_target: str = "Cockroach Janta Party"
    system_prefix: str = (
        "You are an expert social media analyst. Read the comment and question carefully, "
        "then respond with exactly one single letter corresponding to the best option."
    )
    temperature: float = 1.0
    threshold: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if not self.option_keys:
            # Derive clean keys by taking the word before any '('
            self.option_keys = [opt.split("(")[0].strip() for opt in self.options]


def load_tasks(prompts_path: str | Path = "config/prompts.json") -> Dict[str, TaskConfig]:
    """Load task configurations from JSON."""
    p = Path(prompts_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            tasks = {}
            for name, cfg in data.items():
                tasks[name] = TaskConfig(
                    name=name,
                    question=cfg["question"],
                    options=cfg["options"],
                    option_keys=cfg.get("option_keys", []),
                    examples=cfg.get("examples", []),
                    default_target=cfg.get("default_target", "Cockroach Janta Party"),
                    system_prefix=cfg.get("system_prefix", ""),
                )
            return tasks
        except Exception as e:
            logger.warning("Failed to load %s: %s. Using hardcoded defaults.", p, e)

    return {
        "sentiment": TaskConfig(
            name="sentiment",
            question="What is the true underlying sentiment of this comment?",
            options=["positive (sincere praise)", "neutral (factual, question)", "negative (criticism, sarcasm)"],
            option_keys=["positive", "neutral", "negative"],
        ),
        "hate": TaskConfig(
            name="hate",
            question="Does this comment contain hate speech or vulgar abuse?",
            options=["hateful (targeted community hate)", "offensive (vulgar slurs, personal insult)", "neither"],
            option_keys=["hateful", "offensive but not hateful", "neither"],
        ),
        "support": TaskConfig(
            name="support",
            question="What stance does this comment take regarding {target}?",
            options=["supports", "opposes", "neither"],
            option_keys=["supports", "opposes", "neither"],
            default_target="Cockroach Janta Party",
        ),
    }


TASKS = load_tasks()


class CommentClassifier:
    """Classifies comments using one-token letter readout from Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        top_k: int = DEFAULT_TOP_K,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: int = DEFAULT_TIMEOUT,
        keep_alive: Any = DEFAULT_KEEP_ALIVE,
        max_comment_chars: int = MAX_COMMENT_CHARS,
    ):
        self.model = model
        self.url = url
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.max_comment_chars = max_comment_chars
        self._cache: Dict[str, np.ndarray] = {}
        self.session = requests.Session()

    def _cache_key(
        self,
        comment: str,
        task_name: str,
        target: str,
        options: List[str],
        debias: bool,
        post_context: Optional[str] = None,
    ) -> str:
        opts_str = ",".join(options)
        ctx_str = post_context or ""
        raw = f"{self.model}:{task_name}:{target}:{opts_str}:{debias}:{ctx_str}:{comment}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _truncate(self, comment: str) -> str:
        if len(comment) > self.max_comment_chars:
            return comment[: self.max_comment_chars] + "..."
        return comment

    def _build_prompt(
        self,
        comment: str,
        task: TaskConfig,
        target: Optional[str] = None,
        ordered_options: Optional[List[str]] = None,
        post_context: Optional[str] = None,
    ) -> str:
        """
        Build rich prompt with system guidance, options, few-shot examples, and comment.
        """
        options_to_use = ordered_options if ordered_options is not None else task.options
        letters = "ABCDEFGH"[: len(options_to_use)]
        opts_text = "\n".join(f"{l}) {o}" for l, o in zip(letters, options_to_use))

        # Format target placeholder
        q = task.question
        if "{target}" in q:
            target_str = target or task.default_target
            q = q.replace("{target}", target_str)

        # Few-shot examples
        examples_block = ""
        if task.examples and ordered_options is None:
            # Only include static examples when options are in default order
            ex_lines = "\n".join(f"- {ex}" for ex in task.examples)
            examples_block = f"\n\nExamples:\n{ex_lines}"

        context_block = f"\n\nPost Context / Caption: \"{self._truncate(post_context)}\"" if post_context else ""

        prompt = (
            f"{task.system_prefix}"
            f"{context_block}\n\n"
            f"Question: {q}\n"
            f"{opts_text}"
            f"{examples_block}\n\n"
            f"Comment: \"{self._truncate(comment)}\"\n\n"
            f"Answer with one letter only:"
        )
        return prompt

    def _single_pass_probs(
        self,
        comment: str,
        task: TaskConfig,
        target: Optional[str] = None,
        ordered_options: Optional[List[str]] = None,
        post_context: Optional[str] = None,
    ) -> np.ndarray:
        """Single forward pass extracting probability distribution over letters."""
        n_opts = len(ordered_options) if ordered_options is not None else len(task.options)
        letters = "ABCDEFGH"[:n_opts]
        prompt = self._build_prompt(
            comment, task, target=target, ordered_options=ordered_options, post_context=post_context
        )

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": True,
            "top_logprobs": self.top_k,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": 1,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            resp = self.session.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Error communicating with Ollama (%s): %s", self.model, e)
            return np.ones(n_opts) / n_opts

        return self._extract_logprobs(data, letters)

    def _extract_logprobs(self, response: dict, letters: str) -> np.ndarray:
        """Sum logprobs in linear space and renormalize."""
        top = []
        if "logprobs" in response and response["logprobs"]:
            first_pos = response["logprobs"][0]
            top = first_pos.get("top_logprobs") or []

        if not top:
            content = str(response.get("message", {}).get("content") or "").strip().upper()
            probs = np.full(len(letters), 1e-6)
            if content and content[0] in letters:
                probs[letters.index(content[0])] = 1.0
            return probs / probs.sum()

        mass = {l: 0.0 for l in letters}
        for item in top:
            token_str = str(item.get("token") or "").strip().upper()
            if token_str in mass:
                logprob = item.get("logprob")
                if logprob is not None:
                    mass[token_str] += math.exp(float(logprob))

        p = np.array([mass[l] for l in letters], dtype=float) + 1e-9
        return p / p.sum()

    def letter_probs(
        self,
        comment: str,
        task: TaskConfig,
        target: Optional[str] = None,
        debias: bool = False,
        use_cache: bool = True,
        post_context: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract letter probabilities with optional Cyclic Option-Order Debiasing and Post Context.
        """
        cache_key = self._cache_key(
            comment, task.name, target or task.default_target, task.options, debias, post_context=post_context
        )
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        n = len(task.options)
        if not debias or n < 2 or n > 4:
            # Standard single pass (fastest, ~200ms)
            p = self._single_pass_probs(comment, task, target=target, post_context=post_context)
        else:
            # Cyclic Permutation Averaging (3 shifts: completely eliminates position bias)
            accumulated = np.zeros(n, dtype=float)
            shifts = min(3, n)
            for s in range(shifts):
                # Rotate options cyclically by s
                perm_indices = [(i + s) % n for i in range(n)]
                perm_options = [task.options[idx] for idx in perm_indices]

                perm_p = self._single_pass_probs(
                    comment, task, target=target, ordered_options=perm_options, post_context=post_context
                )

                # Map permuted probabilities back to original index space
                for new_idx, orig_idx in enumerate(perm_indices):
                    accumulated[orig_idx] += perm_p[new_idx]

            p = accumulated / shifts

        if use_cache:
            self._cache[cache_key] = p

        return p

    def classify(
        self,
        comment: str,
        task: TaskConfig,
        target: Optional[str] = None,
        debias: bool = False,
        calibration_T: float = 1.0,
        thresholds: Optional[Dict[str, float]] = None,
        post_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify a single comment on a single task."""
        raw_probs = self.letter_probs(comment, task, target=target, debias=debias, post_context=post_context)

        # Temperature calibration
        if calibration_T > 0 and calibration_T != 1.0:
            cal_probs = raw_probs ** (1.0 / calibration_T)
            cal_probs = cal_probs / cal_probs.sum()
        else:
            cal_probs = raw_probs

        pred_idx = int(np.argmax(cal_probs))
        clean_keys = task.option_keys if task.option_keys else task.options
        pred_label = clean_keys[pred_idx]
        confidence = float(cal_probs[pred_idx])

        prob_dict = {clean_keys[i]: float(cal_probs[i]) for i in range(len(clean_keys))}

        needs_review = False
        thresh_dict = thresholds or task.threshold
        if thresh_dict and pred_label in thresh_dict:
            needs_review = confidence < thresh_dict[pred_label]

        return {
            "label": pred_label,
            "confidence": round(confidence, 4),
            "probs": {k: round(v, 4) for k, v in prob_dict.items()},
            "needs_review": needs_review,
        }

    def analyze(
        self,
        comment: str,
        task_names: Optional[List[str]] = None,
        target: Optional[str] = None,
        debias: bool = False,
        custom_tasks: Optional[Dict[str, Dict[str, Any]]] = None,
        temperatures: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        post_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run classification across requested tasks with optional debiasing and post context."""
        if task_names is None and custom_tasks is None:
            task_names = ["sentiment", "hate", "support"]
        elif task_names is None:
            task_names = []

        if temperatures is None:
            temperatures = {}
        if thresholds is None:
            thresholds = {}

        results = {}

        # 1. Built-in tasks
        for name in task_names:
            if name in TASKS:
                task = TASKS[name]
                t_val = temperatures.get(name, task.temperature)
                th_val = thresholds.get(name, task.threshold)
                results[name] = self.classify(
                    comment,
                    task,
                    target=target,
                    debias=debias,
                    calibration_T=t_val,
                    thresholds=th_val,
                    post_context=post_context,
                )

        # 2. Dynamic custom tasks
        if custom_tasks:
            for c_name, c_cfg in custom_tasks.items():
                options = c_cfg.get("options", ["yes", "no"])
                q = c_cfg.get("question", f"Evaluate {c_name}")
                prefix = c_cfg.get("system_prefix", "You are an objective classifier. Answer with one letter only.")

                dyn_task = TaskConfig(
                    name=c_name,
                    question=q,
                    options=options[:8],
                    system_prefix=prefix,
                )
                t_val = temperatures.get(c_name, 1.0)
                th_val = thresholds.get(c_name)
                results[c_name] = self.classify(
                    comment,
                    dyn_task,
                    target=target,
                    debias=debias,
                    calibration_T=t_val,
                    thresholds=th_val,
                    post_context=post_context,
                )

        return results
