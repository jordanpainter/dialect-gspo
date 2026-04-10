#!/usr/bin/env python3
"""
GSPO training status checker.
Queries WandB for active runs and posts a status update to Discord.
"""

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

WANDB_API_KEY = "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"
WANDB_GRAPHQL = "https://api.wandb.ai/graphql"
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1489169370990121111/ENjDEFTJKuIhd_Gnw6l45wteLuhhcYuSux7WbDeRFj__BlZ3n_j2NKsWVTmu-gAa_IPf"

RUNS = [
    ("Gemma", "l3ipzi3b"),
    ("Llama", "dpkjjcv4"),
    ("Qwen",  "qsqszqem"),
]

HISTORY_SPEC = json.dumps({
    "keys": [
        "train/global_step",
        "train/loss",
        "train/kl",
        "train/grad_norm",
        "train/reward",
        "train/train/reward_raw/dialect_gen_mean",
        "train/train/reward_raw/dialect_gain_mean",
        "train/completions/mean_length",
    ],
    "samples": 50,
})


def gql(query: str) -> dict:
    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        WANDB_GRAPHQL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {WANDB_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_run(run_id: str) -> dict:
    spec_escaped = HISTORY_SPEC.replace("\\", "\\\\").replace('"', '\\"')
    query = f'''{{
  project(name: "gspo-all", entityName: "jordanpainter") {{
    run(name: "{run_id}") {{
      state
      summaryMetrics
      sampledHistory(specs: ["{spec_escaped}"])
    }}
  }}
}}'''
    data = gql(query)
    return data["data"]["project"]["run"]


def last_val(rows: list[dict], key: str):
    """Return the last non-None value for key across history rows."""
    for row in reversed(rows):
        v = row.get(key)
        if v is not None:
            return v
    return None


def trend(rows: list[dict], key: str, window: int = 10) -> float | None:
    """Simple linear trend of last `window` non-None values (positive = increasing)."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    if len(vals) < 2:
        return None
    recent = vals[-window:]
    n = len(recent)
    mean_x = (n - 1) / 2
    mean_y = sum(recent) / n
    num = sum((i - mean_x) * (recent[i] - mean_y) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def assess(name: str, run: dict) -> tuple[str, str]:
    """
    Returns (formatted_line, verdict).
    """
    state = (run.get("state") or "unknown").lower()

    if state != "running":
        return (
            f"**{name}** | STOPPED (state={state})",
            "STOPPED",
        )

    # Parse sampledHistory — it's a list of JSON strings, each a list of row dicts
    raw_history = run.get("sampledHistory") or []
    rows: list[dict] = []
    for chunk in raw_history:
        if isinstance(chunk, str):
            rows.extend(json.loads(chunk))
        elif isinstance(chunk, list):
            rows.extend(chunk)

    step        = int(last_val(rows, "train/global_step") or 0)
    reward      = last_val(rows, "train/reward")
    dialect_gen = last_val(rows, "train/train/reward_raw/dialect_gen_mean")
    gain        = last_val(rows, "train/train/reward_raw/dialect_gain_mean")
    kl          = last_val(rows, "train/kl")
    grad        = last_val(rows, "train/grad_norm")
    mean_len    = last_val(rows, "train/completions/mean_length")

    reward_trend = trend(rows, "train/reward")
    gen_trend    = trend(rows, "train/train/reward_raw/dialect_gen_mean")

    # --- Verdict logic ---
    flags = []

    if mean_len is not None and mean_len < 10:
        flags.append("COLLAPSE(len)")
    if kl is not None and kl > 1.0:
        flags.append("CRIT_KL")
    elif kl is not None and kl > 0.5:
        flags.append("HIGH_KL")
    if grad is not None and grad > 100:
        flags.append("HIGH_GRAD")

    if "COLLAPSE(len)" in flags or "CRIT_KL" in flags:
        verdict = "COLLAPSED" if "COLLAPSE(len)" in flags else "UNSTABLE"
    elif "HIGH_KL" in flags or "HIGH_GRAD" in flags:
        verdict = "UNSTABLE"
    elif reward is not None and reward > 0 and (reward_trend or 0) >= 0 and (gen_trend or 0) >= 0:
        verdict = "LEARNING"
    elif step > 0:
        verdict = "HEALTHY"
    else:
        verdict = "UNKNOWN"

    def fmt(v, spec=".3f"):
        return format(v, spec) if v is not None else "N/A"

    sign = "+" if (gain or 0) >= 0 else ""
    line = (
        f"**{name}** | step {step}/5000 | "
        f"reward: {fmt(reward, '.2f')} | "
        f"dialect_gen: {fmt(dialect_gen)} | "
        f"gain: {sign}{fmt(gain)} | "
        f"kl: {fmt(kl)} | "
        f"grad: {fmt(grad, '.1f')} | "
        f"verdict: {verdict}"
    )
    if flags:
        line += f" ⚠ {','.join(flags)}"

    return line, verdict


def overall_recommendation(verdicts: list[str]) -> str:
    if any(v in ("COLLAPSED",) for v in verdicts):
        return "STOP RECOMMENDED"
    if any(v in ("UNSTABLE",) for v in verdicts):
        return "MONITOR"
    return "YES"


def post_discord(message: str):
    body = json.dumps({"content": message}).encode()
    req = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status


def main():
    now = datetime.now(timezone.utc).strftime("%H:%M")

    lines = [f"**GSPO Hourly Check** ({now} UTC)", ""]
    verdicts = []

    for label, run_id in RUNS:
        print(f"Fetching {label} ({run_id})...")
        try:
            run = fetch_run(run_id)
            line, verdict = assess(label, run)
        except Exception as e:
            line = f"**{label}** | ERROR: {e}"
            verdict = "UNKNOWN"
        lines.append(line)
        verdicts.append(verdict)

    lines.append("")
    rec = overall_recommendation(verdicts)
    lines.append(f"**Continue running?** {rec}")

    message = "\n".join(lines)
    print("\n--- Discord message ---")
    print(message)
    print("----------------------")

    print("\nPosting to Discord...")
    status = post_discord(message)
    print(f"Discord response: {status}")


if __name__ == "__main__":
    main()
