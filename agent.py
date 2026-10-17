"""TwinMind Maintenance Assistant: a small tool-using LLM agent.

It answers questions about the saved TwinMind results using four tools that
read files in this repo (no live motor, no invented numbers):

    list_recordings, flagged_summary, model_reliability, search_docs

Put this file in the repo root. Backends (pick one with --backend):
    stub          fake model, for testing the pipeline anywhere (no GPU, no download)
    openai        any OpenAI-compatible server (vLLM, a hosted API): needs --base-url --model
    transformers  a local model folder run with PyTorch/ROCm on the AMD GPU: needs --model-path

Examples:
    python agent.py --backend stub --question "How reliable is the eccentric rotor prediction?"
    python agent.py --backend transformers --model-path /workspace/models/Qwen2.5-3B-Instruct --eval
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SCREENING_CSV = HERE / "reports" / "amd_gpu" / "twinmind_gpu_sensor_screening.csv"
REPORT_CSV = HERE / "reports" / "temporal_gpu" / "classification_report.csv"
CONFUSION_CSV = HERE / "reports" / "temporal_gpu" / "confusion_matrix.csv"
DOCS_DIR = HERE / "maintenance_docs"
EVAL_OUT = HERE / "reports" / "agent" / "agent_eval.json"

DISCLAIMER = (
    "Decision support only: a qualified technician must verify any suspected "
    "fault before maintenance decisions are made."
)
EVAL_NOTE = (
    "chronological holdout on one file; shows temporal generalisation, "
    "not performance on a new motor"
)


# ----------------------------------------------------------------- tools
def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def list_recordings():
    df = pd.read_csv(SCREENING_CSV)
    return {"recordings": sorted(df["source_file"].unique().tolist())}


def flagged_summary(recording):
    df = pd.read_csv(SCREENING_CSV)
    names = df["source_file"].unique().tolist()
    key = _norm(recording)
    hits = [n for n in names if key and key in _norm(n)]
    if not hits:
        return {"error": f"No recording matches '{recording}'.", "available": sorted(names)}
    name = sorted(hits, key=len)[0]
    part = df[df["source_file"] == name]
    flagged = part["flagged_for_review"].astype(str).str.lower().isin(["true", "1", "yes"])
    top = part.nlargest(3, "screening_score")[["source_row", "screening_score"]].round(4)
    return {
        "recording": name,
        "rows": int(len(part)),
        "flagged": int(flagged.sum()),
        "flagged_percent": round(100.0 * float(flagged.mean()), 2),
        "mean_screening_score": round(float(part["screening_score"].mean()), 4),
        "top_scores": top.to_dict("records"),
        "note": "Flags rows above the dataset's 95th percentile; a screening aid, not a fault diagnosis.",
    }


def model_reliability(condition="all"):
    report = pd.read_csv(REPORT_CSV, index_col=0)
    matrix = pd.read_csv(CONFUSION_CSV).set_index("true_status")
    classes = list(matrix.index)

    def one(name):
        row = report.loc[name]
        wrong = {p: int(v) for p, v in matrix.loc[name].items() if p != name and int(v) > 0}
        return {
            "condition": name,
            "precision": round(float(row["precision"]), 3),
            "recall": round(float(row["recall"]), 3),
            "f1": round(float(row["f1-score"]), 3),
            "test_records": int(row["support"]),
            "misclassified_as": wrong,
        }

    key = re.sub(r"[\s-]+", "_", str(condition).strip().lower())
    if key in ("all", "", "any"):
        rows = [one(c) for c in classes]
        return {
            "classes": rows,
            "weakest_recall": min(rows, key=lambda d: d["recall"])["condition"],
            "accuracy": round(float(report.loc["accuracy", "precision"]), 4),
            "evaluation": EVAL_NOTE,
        }
    if key not in classes:
        return {"error": f"Unknown condition '{condition}'.", "available": classes}
    result = one(key)
    result["evaluation"] = EVAL_NOTE
    return result


def search_docs(query, k=2):
    """Retrieve maintenance passages while preventing cross-fault matches."""
    words = {
        w for w in re.findall(r"[a-z0-9]+", str(query).lower())
        if len(w) > 2
    }

    topic_groups = {
        "bearing": {"bearing", "bearings"},
        "shaft": {"shaft", "shafts"},
        "rotor": {"rotor", "rotors"},
        "coupling": {"coupling", "couplings"},
        "imbalance": {"imbalance", "unbalance"},
    }
    query_topics = {
        topic for topic, variants in topic_groups.items()
        if words & variants
    }

    scored = []
    if DOCS_DIR.is_dir():
        for path in sorted(DOCS_DIR.glob("*.txt")):
            text = path.read_text(
                encoding="utf-8", errors="replace"
            ).lstrip("\\ufeff")
            chunks = [
                c.strip()
                for c in re.split(r"\\n\\s*\\n", text)
                if c.strip()
            ] or [text.strip()]

            for chunk in chunks:
                searchable_text = (
                    path.stem.replace("_", " ").lower()
                    + " " + chunk.lower()
                )
                doc_words = set(
                    re.findall(r"[a-z0-9]+", searchable_text)
                )

                # If a fault type is named, require matching topic evidence.
                if query_topics and not any(
                    topic in doc_words for topic in query_topics
                ):
                    continue

                score = len(words & doc_words)
                if score:
                    scored.append((score, path.name, chunk[:600]))

    scored.sort(key=lambda item: -item[0])
    if not scored:
        if query_topics:
            topic = ", ".join(sorted(query_topics))
            note = (
                f"No relevant maintenance guidance for {topic} "
                "was found in the available documents."
            )
        else:
            note = "No matching passage in the maintenance documents."
        return {"matches": [], "note": note}

    return {
        "matches": [
            {"file": filename, "passage": passage}
            for score, filename, passage in scored[:int(k)]
        ]
    }
def _parse_json(text):
    """First JSON object in the text (tolerates ```json fences and chatter)."""
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        start = text.find("{", start + 1)
    return None


def run_agent(question, llm, max_steps=4):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    used, doc_files, answer = [], [], None
    begin = time.perf_counter()
    for _ in range(max_steps):
        raw = llm.chat(messages)
        obj = _parse_json(raw)
        if obj is None:
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Reply with exactly one JSON object as instructed."},
            ]
            continue
        if "final" in obj:
            answer = str(obj["final"])
            break
        name, args = obj.get("tool"), obj.get("args") or {}
        if name in TOOLS:
            try:
                result = TOOLS[name][0](**args)
            except Exception as exc:  # report the problem to the model, do not crash
                result = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            result = {"error": f"unknown tool '{name}'", "tools": list(TOOLS)}
        used.append({"tool": name, "args": args, "result": result})
        if isinstance(result, dict):
            doc_files += [m["file"] for m in result.get("matches", [])]
        messages += [
            {"role": "assistant", "content": json.dumps(obj)},
            {"role": "user", "content": "Tool result: " + json.dumps(result)[:3000]},
        ]
    if answer is None:
        answer = "I could not finish within the step limit."

    # Sources and the disclaimer are added by code, so they cannot be invented.
    calls = "; ".join(f"{u['tool']}({json.dumps(u['args'])})" for u in used) or "none"
    docs = f" | documents: {', '.join(sorted(set(doc_files)))}" if doc_files else ""
    return {
        "question": question,
        "answer": f"{answer}\n\nSources: tools used: {calls}{docs}\n{DISCLAIMER}",
        "tools_used": [u["tool"] for u in used],
        "trace": used,
        "backend": getattr(llm, "name", "unknown"),
        "seconds": round(time.perf_counter() - begin, 2),
    }


_CONDITION_HINTS = (
    ("eccentric", "eccentric_rotor"),
    ("bent", "bent_shaft"),
    ("bearing", "faulted_bearing"),
    ("imbalance", "imbalance"),
    ("baseline", "baseline"),
)


def _pick_condition(question):
    lowered = question.lower()
    for hint, name in _CONDITION_HINTS:
        if hint in lowered:
            return name
    return "all"


def _pick_recording(question, hint=""):
    match = re.search(r"(Combined[_-][\w]+\.csv)", question, re.I)
    if match:
        return match.group(1)
    return str(hint or "").strip()


def _invoke_tool(name, **kwargs):
    try:
        result = TOOLS[name][0](**kwargs)
    except Exception as exc:  # noqa: BLE001
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return {"tool": name, "args": kwargs, "result": result}


def _compose_grounded_answer(question, steps, confirm):
    parts = []
    if confirm:
        parts.append(
            "No. TwinMind cannot confirm that a motor has a physical fault. "
            "Use the figures below as a screening brief for a technician, not as a diagnosis."
        )

    by_name = {step["tool"]: step["result"] for step in steps}

    listed = by_name.get("list_recordings") or {}
    if listed.get("recordings"):
        names = listed["recordings"]
        preview = ", ".join(names[:8])
        extra = f" and {len(names) - 8} more" if len(names) > 8 else ""
        parts.append(f"Saved AMD screening results cover {len(names)} recordings{extra}: {preview}.")

    flagged = by_name.get("flagged_summary") or {}
    if flagged.get("error"):
        parts.append(str(flagged["error"]))
    elif flagged.get("recording"):
        parts.append(
            f"{flagged['recording']}: {flagged['flagged']:,} of {flagged['rows']:,} rows "
            f"({flagged['flagged_percent']}%) were flagged for review. "
            f"Mean screening score {flagged['mean_screening_score']}. "
            f"{flagged.get('note', '')}"
        )
        top = flagged.get("top_scores") or []
        if top:
            top_text = "; ".join(
                f"row {item.get('source_row')} score {item.get('screening_score')}"
                for item in top
            )
            parts.append(f"Highest scores: {top_text}.")

    reliability = by_name.get("model_reliability") or {}
    if reliability.get("error"):
        parts.append(str(reliability["error"]))
    elif reliability.get("classes"):
        weakest = reliability.get("weakest_recall")
        parts.append(
            f"Holdout accuracy {reliability.get('accuracy')}. "
            f"Weakest recall: {weakest}. {reliability.get('evaluation', '')}"
        )
    elif reliability.get("condition"):
        parts.append(
            f"{reliability['condition']}: precision {reliability['precision']}, "
            f"recall {reliability['recall']}, F1 {reliability['f1']} "
            f"on {reliability['test_records']} holdout records. "
            f"{reliability.get('evaluation', '')}"
        )
        mistakes = reliability.get("misclassified_as") or {}
        if mistakes:
            parts.append("Typical mistakes: " + ", ".join(f"{k} ({v})" for k, v in mistakes.items()) + ".")

    docs = by_name.get("search_docs") or {}
    matches = docs.get("matches") or []
    if matches:
        parts.append("Inspection notes from the maintenance documents:")
        for match in matches:
            parts.append(f"- {match['file']}: {match['passage']}")
    elif "search_docs" in by_name:
        parts.append(docs.get("note") or "No matching maintenance passage.")

    if not parts:
        parts.append(
            "I can summarise flagged rows, classifier reliability, or inspection notes "
            "from the saved TwinMind files. Ask about a recording, a condition, or what to check next."
        )

    calls = "; ".join(
        f"{step['tool']}({json.dumps(step['args'])})" for step in steps
    ) or "none"
    parts.append(f"Sources: tools used: {calls}")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)



def _is_unanswerable_cause_question(question):
    """Recognize cause questions the saved data cannot answer."""
    text = str(question).lower()

    asks_why = bool(
        re.search(r"\bwhy\b|\bwhat caused\b|\broot cause\b", text)
    )
    mentions_fault = any(
        word in text
        for word in (
            "motor", "fault", "faulty", "bearing",
            "shaft", "rotor", "coupling", "imbalance",
        )
    )
    asks_for_evidence = any(
        word in text
        for word in (
            "flag", "screen", "score", "precision", "recall",
            "accuracy", "f1", "how many", "rows", "check",
            "inspect", "inspection", "maintenance",
        )
    )

    return asks_why and mentions_fault and not asks_for_evidence

def plan_and_answer(question, recording_hint=""):
    """Answer from the repo files with a keyword planner (no invented numbers)."""
    begin = time.perf_counter()
    lowered = str(question).lower()
    recording = _pick_recording(question, recording_hint)
    if _is_unanswerable_cause_question(question):
        return {
            "question": question,
            "answer": (
                "I don't know the exact cause of the motor fault "
                "based on the available information. TwinMind's screening "
                "flags unusual readings, but those flags alone cannot "
                "confirm a physical fault or establish its cause. "
                "The motor and sensor readings need to be verified by "
                "a qualified technician.\n\n"
                + DISCLAIMER
            ),
            "tools_used": [],
            "trace": [],
            "backend": "grounded tools (saved AMD reports)",
            "seconds": round(time.perf_counter() - begin, 2),
        }
    confirm = any(
        token in lowered
        for token in ("definitely", "confirm this", "guaranteed", "prove that", "diagnos")
    )
    wants_list = any(
        token in lowered
        for token in ("which recording", "what recording", "available", "have results")
    )
    wants_flag = any(
        token in lowered
        for token in (
            "flag",
            "screen",
            "briefing",
            "how many",
            "review",
            "this recording",
            "combined_",
            "inspect",
            "draft",
        )
    )
    wants_rel = any(
        token in lowered
        for token in ("reliab", "precision", "recall", "wrong", "accuracy", "f1", "misclass")
    )
    wants_docs = any(
        token in lowered
        for token in (
            "inspect",
            "check",
            "bearing",
            "shaft",
            "imbalance",
            "maintenance",
            "what should",
            "action",
            "draft",
            "briefing",
        )
    )

    planned = []
    if wants_list:
        planned.append(("list_recordings", {}))
    if wants_flag and recording:
        planned.append(("flagged_summary", {"recording": recording}))
    elif wants_flag and not recording and not wants_list:
        planned.append(("list_recordings", {}))
    if wants_rel:
        planned.append(("model_reliability", {"condition": _pick_condition(question)}))
    if wants_docs:
        planned.append(("search_docs", {"query": question}))
    if not planned and recording:
        planned.append(("flagged_summary", {"recording": recording}))
        planned.append(("search_docs", {"query": question}))
    if not planned:
        planned.append(("list_recordings", {}))

    # Preserve order, drop duplicate tool names so a briefing stays short.
    seen, steps = set(), []
    for name, args in planned:
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        steps.append(_invoke_tool(name, **args))

    return {
        "question": question,
        "answer": _compose_grounded_answer(question, steps, confirm),
        "tools_used": [step["tool"] for step in steps],
        "trace": steps,
        "backend": "grounded tools (saved AMD reports)",
        "seconds": round(time.perf_counter() - begin, 2),
    }


# -------------------------------------------------------------- evaluation
QUESTIONS = [
    ("How many rows of Combined_BentShaft.csv were flagged for review?", {"flagged_summary"}),
    ("Does the screening flag anything in Combined_Baseline1.csv?", {"flagged_summary"}),
    ("How reliable is the eccentric rotor prediction?", {"model_reliability"}),
    ("Which condition does the classifier get wrong most often?", {"model_reliability"}),
    ("What should I check first for a faulted bearing?", {"search_docs"}),
    ("What vibration checks should I do before escalating?", {"search_docs"}),
    ("Give me a short briefing for Combined_BentShaft.csv: flagged share and what to inspect.", {"flagged_summary", "search_docs"}),
    ("Which recordings do you have results for?", {"list_recordings"}),
    ("Can you confirm this motor definitely has a bearing fault?", set()),
    ("Draft an inspection note for Combined_FaultedBearing.csv.", {"flagged_summary", "search_docs"}),
]


def run_eval(llm):
    rows = []
    for question, expected in QUESTIONS:
        result = run_agent(question, llm)
        result["expected_tools"] = sorted(expected)
        result["called_expected_tools"] = expected <= set(result["tools_used"])
        rows.append(result)
        print(f"[{'ok' if result['called_expected_tools'] else 'MISS'}] {question}\n"
              f"      tools: {result['tools_used']}  ({result['seconds']} s)")
    summary = {
        "model": llm.name,
        "questions": len(rows),
        "called_expected_tools": sum(r["called_expected_tools"] for r in rows),
        "tokens_per_second": round(llm.new_tokens / llm.gen_seconds, 1) if llm.gen_seconds else None,
        "note": "Answers must be read by a person; 'called_expected_tools' only checks tool use.",
    }
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.write_text(json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
    print("\n", summary, "\nSaved", EVAL_OUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["stub", "openai", "transformers"], default="stub")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="not-required")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--eval", action="store_true", help="run the 10 test questions")
    args = parser.parse_args()

    if args.backend == "openai":
        llm = OpenAICompatLLM(args.base_url, args.model, args.api_key)
    elif args.backend == "transformers":
        llm = TransformersLLM(args.model_path)
    else:
        llm = StubLLM()
    print("Model:", llm.name)

    if args.eval:
        run_eval(llm)
    elif args.question:
        result = run_agent(args.question, llm)
        print("\n" + result["answer"])
    else:
        parser.error("give --question or --eval")


if __name__ == "__main__":
    main()

