#!/usr/bin/env python3
"""
GSPO hourly status check: queries WandB for all three runs, analyzes metrics,
and posts a formatted summary to Discord.
"""

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

WANDB_API_KEY = "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"
WANDB_GRAPHQL = "https://api.wandb.ai/graphql"
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1489169370990121111/ENjDEFTJKuIhd_Gnw6l45wteLuhhcYuSux7WbDeRFj__BlZ3n_j2NKsWVTmu-gAa_IPf"

RUNS = {
    "Gemma": "l3ipzi3b",
    "Llama": "dpkjjcv4",
    "Qwen":  "qsqszqem",
}

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

QUERY_TEMPLATE = """
{{
  project(name: "gspo-all", entityName: "jordanpainter") {{
    run(name: "{run_id}") {{
      state
      summaryMetrics
      sampledHistory(specs: ["{spec}"])
    }}
  }}
}}
"""


def wandb_query(run_id: str) -> dict:
    spec_escaped = HISTORY_SPEC.replace("\\", "\\\\").replace('"', '\\"')
    query = QUERY_TEMPLATE.format(run_id=run_id, spec=spec_escaped)
    payload = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        WANDB_GRAPHQL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {WANDB_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def last_val(history: list[dict], key: str):
    """Return the most recent non-None value for a key across history rows."""
    for row in reversed(history):
        v = row.get(key)
        if v is not None:
            return v
    return None


def trend(history: list[dict], key: str, window: int = 5) -> str:
    """Return '↑', '↓', or '→' based on last `window` values."""
    vals = [r[key] for r in history if r.get(key) is not None]
    if len(vals) < 2:
        return "?"
    recent = vals[-window:]
    if recent[-1] > recent[0] + 1e-6:
        return "↑"
    elif recent[-1] < recent[0] - 1e-6:
        return "↓"
    return "→"


def assess(run_name: str, state: str, history: list[dict]) -> dict:
    step        = last_val(history, "train/global_step") or 0
    reward      = last_val(history, "train/reward")
    dialect_gen = last_val(history, "train/train/reward_raw/dialect_gen_mean")
    dialect_gain= last_val(history, "train/train/reward_raw/dialect_gain_mean")
    kl          = last_val(history, "train/kl")
    grad_norm   = last_val(history, "train/grad_norm")
    mean_len    = last_val(history, "train/completions/mean_length")

    flags = []
    if kl is not None and kl > 1.0:
        flags.append("CRITICAL_KL")
    elif kl is not None and kl > 0.5:
        flags.append("HIGH_KL")
    if grad_norm is not None and grad_norm > 100:
        flags.append("HIGH_GRAD")
    if mean_len is not None and mean_len < 10:
        flags.append("COLLAPSE")

    reward_trend = trend(history, "train/reward")
    gen_trend    = trend(history, "train/train/reward_raw/dialect_gen_mean")

    if state != "running":
        verdict = "STOPPED"
    elif "COLLAPSE" in flags:
        verdict = "COLLAPSED"
    elif "CRITICAL_KL" in flags or ("HIGH_KL" in flags and "HIGH_GRAD" in flags):
        verdict = "UNSTABLE"
    elif reward is not None and reward > 0 and reward_trend in ("↑", "→") and gen_trend == "↑":
        verdict = "LEARNING"
    elif reward is not None and reward > 0:
        verdict = "HEALTHY"
    else:
        verdict = "UNSTABLE"

    return {
        "name": run_name,
        "state": state,
        "step": int(step),
        "reward": reward,
        "dialect_gen": dialect_gen,
        "dialect_gain": dialect_gain,
        "kl": kl,
        "grad_norm": grad_norm,
        "mean_len": mean_len,
        "flags": flags,
        "verdict": verdict,
        "reward_trend": reward_trend,
        "gen_trend": gen_trend,
    }


def fmt_val(v, decimals=3):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def build_message(assessments: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M")
    lines = [f"**GSPO Hourly Check** ({now} UTC)", ""]

    verdicts = [a["verdict"] for a in assessments]
    if any(v in ("COLLAPSED", "STOPPED") for v in verdicts):
        overall = "STOP RECOMMENDED"
    elif any(v == "UNSTABLE" for v in verdicts):
        overall = "MONITOR"
    else:
        overall = "YES"

    for a in assessments:
        gain_str = fmt_val(a["dialect_gain"])
        if a["dialect_gain"] is not None and a["dialect_gain"] >= 0:
            gain_str = "+" + gain_str

        flag_str = ""
        if a["flags"]:
            flag_str = " ⚠ " + ",".join(a["flags"])

        state_note = "" if a["state"] == "running" else f" [STATE={a['state'].upper()}]"

        lines.append(
            f"**{a['name']}** | step {a['step']}/5000 | "
            f"reward: {fmt_val(a['reward'], 2)} | "
            f"dialect_gen: {fmt_val(a['dialect_gen'])} | "
            f"gain: {gain_str} | "
            f"kl: {fmt_val(a['kl'])} | "
            f"grad: {fmt_val(a['grad_norm'], 1)} | "
            f"verdict: {a['verdict']}{state_note}{flag_str}"
        )

    lines += ["", f"**Continue running?** {overall}"]
    return "\n".join(lines)


def post_discord(message: str):
    payload = json.dumps({"content": message}).encode()
    req = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status


def main():
    assessments = []
    for name, run_id in RUNS.items():
        print(f"Fetching {name} ({run_id})...")
        try:
            data = wandb_query(run_id)
            run = data["data"]["project"]["run"]
            state = run["state"]
            raw_history = run["sampledHistory"]
            # sampledHistory is a list of lists of row-dicts; flatten to single list
            history = []
            for chunk in raw_history:
                if isinstance(chunk, list):
                    history.extend(chunk)
                elif isinstance(chunk, dict):
                    history.append(chunk)
            a = assess(name, state, history)
        except Exception as e:
            print(f"  ERROR: {e}")
            a = {
                "name": name, "state": "error", "step": 0,
                "reward": None, "dialect_gen": None, "dialect_gain": None,
                "kl": None, "grad_norm": None, "mean_len": None,
                "flags": ["FETCH_ERROR"], "verdict": "UNSTABLE",
                "reward_trend": "?", "gen_trend": "?",
            }
        assessments.append(a)
        print(f"  step={a['step']} reward={a['reward']} verdict={a['verdict']}")

    message = build_message(assessments)
    print("\n--- Discord message ---")
    print(message)
    print("----------------------\n")

    print("Posting to Discord...")
    status = post_discord(message)
    print(f"Discord response status: {status}")


if __name__ == "__main__":
    main()
