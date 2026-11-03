"""
Desktop GUI for Social Media Comment Analysis (Sentiment · Hate · Stance · Custom Tasks).

Usage:
    python gui.py
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
import requests

API_URL = "http://127.0.0.1:8000/analyze"


class CommentClassifierApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MOJ — Comment Classifier (Qwen3.5-9B & DeepSeek)")
        self.root.geometry("880x700")
        self.root.minsize(760, 600)
        self.root.configure(bg="#0f172a")

        self.classifier = None
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._configure_styles()

        self._build_ui()

    def _configure_styles(self):
        self.style.configure(".", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 10))
        self.style.configure("TLabel", background="#0f172a", foreground="#f8fafc")
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#6366f1")
        self.style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#94a3b8")
        self.style.configure("Card.TFrame", background="#1e293b", relief="flat")
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, background="#4f46e5", foreground="#ffffff")
        self.style.map("TButton", background=[("active", "#6366f1")])

    def _build_ui(self):
        # Header
        header_frame = ttk.Frame(self.root, padding="16 12 16 8")
        header_frame.pack(fill="x")
        
        top_bar = ttk.Frame(header_frame)
        top_bar.pack(fill="x")
        
        left_h = ttk.Frame(top_bar)
        left_h.pack(side="left")
        ttk.Label(left_h, text="⚡ MOJ · Comment Classifier (Qwen3.5-9B)", style="Header.TLabel").pack(anchor="w")
        ttk.Label(left_h, text="Calibrated One-Token Readout · Generalized for any campaign or topic", style="Sub.TLabel").pack(anchor="w")

        # Right model selection
        right_h = ttk.Frame(top_bar)
        right_h.pack(side="right")
        ttk.Label(right_h, text="Model:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)
        self.model_var = tk.StringVar(value="qwen3.5-judge")
        model_combo = ttk.Combobox(right_h, textvariable=self.model_var, values=["qwen3.5-judge", "comment-judge"], width=15, state="readonly")
        model_combo.pack(side="left")

        # Config / Target Row
        cfg_frame = tk.Frame(self.root, bg="#1e293b", bd=1, relief="solid", padx=12, pady=8)
        cfg_frame.pack(fill="x", padx=16, pady=4)

        # Stance Target Input
        tk.Label(cfg_frame, text="Stance Target Entity:", font=("Segoe UI", 9, "bold"), bg="#1e293b", fg="#94a3b8").pack(side="left", padx=(0, 6))
        self.target_entry = tk.Entry(cfg_frame, bg="#020617", fg="#f8fafc", insertbackground="#6366f1", font=("Segoe UI", 9), width=24)
        self.target_entry.insert(0, "Cockroach Janta Party")
        self.target_entry.pack(side="left", padx=(0, 12))

        # Post Context Input
        tk.Label(cfg_frame, text="Post Caption/Context:", font=("Segoe UI", 9, "bold"), bg="#1e293b", fg="#94a3b8").pack(side="left", padx=(0, 6))
        self.post_context_entry = tk.Entry(cfg_frame, bg="#020617", fg="#f8fafc", insertbackground="#6366f1", font=("Segoe UI", 9), width=28)
        self.post_context_entry.pack(side="left", padx=(0, 12))

        # Sentiment Granularity Option
        tk.Label(cfg_frame, text="Sentiment:", font=("Segoe UI", 9, "bold"), bg="#1e293b", fg="#94a3b8").pack(side="left", padx=(0, 6))
        self.sent_mode_var = tk.StringVar(value="Standard (3-class)")
        sent_combo = ttk.Combobox(cfg_frame, textvariable=self.sent_mode_var, values=["Standard (3-class)", "Fine-Grained + Sarcasm (5-class)"], width=22, state="readonly")
        sent_combo.pack(side="left")

        # Input Frame
        input_frame = ttk.Frame(self.root, padding="16 8 16 8")
        input_frame.pack(fill="x")

        ttk.Label(input_frame, text="Insert Comment:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        
        self.text_input = tk.Text(
            input_frame,
            height=3,
            bg="#020617",
            fg="#f8fafc",
            insertbackground="#6366f1",
            relief="solid",
            bd=1,
            font=("Segoe UI", 10),
            wrap="word",
        )
        self.text_input.pack(fill="x", pady=(0, 6))

        # Quick sample buttons
        sample_frame = ttk.Frame(input_frame)
        sample_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(sample_frame, text="Quick Samples:", font=("Segoe UI", 9, "italic"), foreground="#94a3b8").pack(side="left", padx=(0, 6))

        samples = [
            ("Support CJP", "Full support to CJP team! Adivasi school repair must happen immediately 🙌", "Cockroach Janta Party"),
            ("Resign Meme", "Narendra Modi resign button 👇", "Narendra Modi"),
            ("Abusive Hindi", "ye bta bhadwe tmko kaun sa post mili bartan saf krne ka", "Cockroach Janta Party"),
            ("Sarcasm", "Great job destroying the economy, truly masterclass 👏", "Government Policy"),
        ]
        for name, txt, tgt in samples:
            btn = tk.Button(
                sample_frame,
                text=name,
                font=("Segoe UI", 8),
                bg="#334155",
                fg="#f8fafc",
                activebackground="#475569",
                activeforeground="#ffffff",
                relief="flat",
                command=lambda t=txt, tg=tgt: self._set_text(t, tg),
            )
            btn.pack(side="left", padx=3)

        # Action Button & Status
        action_frame = ttk.Frame(input_frame)
        action_frame.pack(fill="x")

        self.analyze_btn = tk.Button(
            action_frame,
            text="Analyze Comment",
            font=("Segoe UI", 10, "bold"),
            bg="#4f46e5",
            fg="#ffffff",
            activebackground="#6366f1",
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=6,
            command=self.on_analyze_click,
        )
        self.analyze_btn.pack(side="left")

        # Debiasing Checkbox
        self.debias_var = tk.BooleanVar(value=False)
        self.debias_check = tk.Checkbutton(
            action_frame,
            text="🛡️ Cyclic Debiasing (removes position bias)",
            variable=self.debias_var,
            bg="#0f172a",
            fg="#fbbf24",
            selectcolor="#1e293b",
            activebackground="#0f172a",
            activeforeground="#f59e0b",
            font=("Segoe UI", 9),
        )
        self.debias_check.pack(side="left", padx=12)

        self.status_label = ttk.Label(action_frame, text="Ready (Model: Qwen3.5-9B)", style="Sub.TLabel")
        self.status_label.pack(side="left", padx=12)

        # Results Cards Frame
        self.results_frame = ttk.Frame(self.root, padding="16 12 16 16")
        self.results_frame.pack(fill="both", expand=True)

        self.cards_container = ttk.Frame(self.results_frame)
        self.cards_container.pack(fill="both", expand=True)

        # Pre-create standard 3 cards
        self.cards = {}
        self.cards["sentiment"] = self._create_card(self.cards_container, "SENTIMENT", 0)
        self.cards["hate"] = self._create_card(self.cards_container, "HATE & ABUSE", 1)
        self.cards["support"] = self._create_card(self.cards_container, "STANCE / TARGET", 2)

    def _create_card(self, parent, title: str, col: int) -> dict:
        parent.columnconfigure(col, weight=1, uniform="group1")
        card = tk.Frame(parent, bg="#1e293b", bd=1, relief="solid", padx=12, pady=12)
        card.grid(row=0, column=col, sticky="nsew", padx=6, pady=4)

        t_lbl = tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), bg="#1e293b", fg="#94a3b8")
        t_lbl.pack(anchor="w")

        badge = tk.Label(card, text="--", font=("Segoe UI", 13, "bold"), bg="#1e293b", fg="#e2e8f0")
        badge.pack(anchor="w", pady=(8, 2))

        conf_lbl = tk.Label(card, text="Confidence: --%", font=("Segoe UI", 9), bg="#1e293b", fg="#94a3b8")
        conf_lbl.pack(anchor="w")

        sep = tk.Frame(card, height=1, bg="#334155")
        sep.pack(fill="x", pady=8)

        details = tk.Label(card, text="", font=("Consolas", 8), bg="#1e293b", fg="#cbd5e1", justify="left")
        details.pack(anchor="w", fill="x")

        return {"card": card, "t_lbl": t_lbl, "badge": badge, "conf": conf_lbl, "details": details}

    def _set_text(self, text: str, target: str):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", text)
        self.target_entry.delete(0, tk.END)
        self.target_entry.insert(0, target)
        self.on_analyze_click()

    def on_analyze_click(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty Comment", "Please type or paste a comment first.")
            return

        target = self.target_entry.get().strip() or "the campaign / issue"
        post_context = self.post_context_entry.get().strip() or None
        sent_mode = self.sent_mode_var.get()
        task_name = "sentiment_fine" if "Fine" in sent_mode else "sentiment"
        model = self.model_var.get()
        debias = self.debias_var.get()

        self.analyze_btn.config(state="disabled")
        status_suffix = " (with 3-pass debiasing)" if debias else ""
        self.status_label.config(text=f"Analyzing with {model}{status_suffix}...")

        threading.Thread(
            target=self._run_inference,
            args=(text, sent_task, target, model, debias, post_context),
            daemon=True,
        ).start()

    def _run_inference(self, text: str, sent_task: str, target: str, model: str, debias: bool, post_context: str = None):
        payload = {
            "text": text,
            "tasks": [sent_task, "hate", "support"],
            "target": target,
            "post_context": post_context,
            "model": model,
            "debias": debias,
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                self.root.after(0, self._render_results, data["results"], data.get("elapsed_ms"), sent_task, target)
                return
        except Exception:
            pass

        # Fallback to direct python call
        try:
            from src.readout import CommentClassifier
            if self.classifier is None or self.classifier.model != model:
                self.classifier = CommentClassifier(model=model)
            res = self.classifier.analyze(
                text, task_names=[sent_task, "hate", "support"], target=target, debias=debias, post_context=post_context
            )
            self.root.after(0, self._render_results, res, None, sent_task, target)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _render_results(self, res: dict, elapsed_ms: float | None, sent_task: str, target: str):
        self.analyze_btn.config(state="normal")
        status_text = "Done" if elapsed_ms is None else f"Analyzed in {elapsed_ms:.0f} ms ({self.model_var.get()})"
        self.status_label.config(text=status_text)

        # 1. Sentiment Card
        s = res.get(sent_task, {})
        s_label = s.get("label", "--").upper()
        s_conf = s.get("confidence", 0) * 100
        self.cards["sentiment"]["t_lbl"].config(text=f"SENTIMENT ({sent_task.upper()})")
        self.cards["sentiment"]["badge"].config(text=s_label)
        if "POSITIVE" in s_label:
            self.cards["sentiment"]["badge"].config(fg="#34d399")
        elif "NEGATIVE" in s_label:
            self.cards["sentiment"]["badge"].config(fg="#f87171")
        elif "SARCASTIC" in s_label:
            self.cards["sentiment"]["badge"].config(fg="#fbbf24")
        else:
            self.cards["sentiment"]["badge"].config(fg="#94a3b8")
        self.cards["sentiment"]["conf"].config(text=f"Confidence: {s_conf:.1f}%")
        self.cards["sentiment"]["details"].config(text=self._format_probs(s.get("probs", {})))

        # 2. Hate Card
        h = res.get("hate", {})
        h_label = h.get("label", "--").upper()
        h_conf = h.get("confidence", 0) * 100
        self.cards["hate"]["badge"].config(text=h_label)
        if h_label == "HATEFUL":
            self.cards["hate"]["badge"].config(fg="#f87171")
        elif "OFFENSIVE" in h_label:
            self.cards["hate"]["badge"].config(fg="#fbbf24")
        else:
            self.cards["hate"]["badge"].config(fg="#34d399")
        flag_str = " [FLAGGED]" if h.get("flag") else ""
        self.cards["hate"]["conf"].config(text=f"Confidence: {h_conf:.1f}%{flag_str}")
        self.cards["hate"]["details"].config(text=self._format_probs(h.get("probs", {})))

        # 3. Support Card
        st = res.get("support", {})
        st_label = st.get("label", "--").upper()
        st_conf = st.get("confidence", 0) * 100
        self.cards["support"]["t_lbl"].config(text=f"STANCE ON: {target[:20].upper()}")
        self.cards["support"]["badge"].config(text=st_label)
        if st_label == "SUPPORTS":
            self.cards["support"]["badge"].config(fg="#818cf8")
        elif st_label == "OPPOSES":
            self.cards["support"]["badge"].config(fg="#c084fc")
        else:
            self.cards["support"]["badge"].config(fg="#94a3b8")
        self.cards["support"]["conf"].config(text=f"Confidence: {st_conf:.1f}%")
        self.cards["support"]["details"].config(text=self._format_probs(st.get("probs", {})))

    def _format_probs(self, probs: dict) -> str:
        lines = []
        for k, v in probs.items():
            pct = v * 100
            bar = "■" * int(pct // 10) + "□" * (10 - int(pct // 10))
            lines.append(f"{k[:11]:<11} {bar} {pct:4.1f}%")
        return "\n".join(lines)

    def _show_error(self, err: str):
        self.analyze_btn.config(state="normal")
        self.status_label.config(text="Error occurred")
        messagebox.showerror("Inference Error", err)


if __name__ == "__main__":
    root = tk.Tk()
    app = CommentClassifierApp(root)
    root.mainloop()
