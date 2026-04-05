#!/usr/bin/env python3
"""GSPO hourly check — queries WandB and posts to Discord."""
import json, urllib.request, urllib.error
from datetime import datetime, timezone

API_KEY = "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"
WANDB_URL = "https://api.wandb.ai/graphql"
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1489169370990121111/ENjDEFTJKuIhd_Gnw6l45wteLuhhcYuSux7WbDeRFj__BlZ3n_j2NKsWVTmu-gAa_IPf"

RUNS = {
    "Gemma": "l3ipzi3b",
    "Llama": "dpkjjcv4",
    "Qwen":  "qsqszqem",
}

SPEC = json.dumps({
    "keys": [
        "train/global_step", "train/loss", "train/kl", "train/grad_norm",
        "train/reward", "train/train/reward_raw/dialect_gen_mean",
        "train/train/reward_raw/dialect_gain_mean", "train/completions/mean_length"
    ],
    "samples": 50
})

QUERY = '''{{
  project(name: "gspo-all", entityName: "jordanpainter") {{
    run(name: "{run_id}") {{
      state
      summaryMetrics
      sampledHistory(specs: [{spec}])
    }}
  }}
}}'''


def gql(run_id):
    q = QUERY.format(run_id=run_id, spec=json.dumps(SPEC))
    data = json.dumps({"query": q}).encode()
    req = urllib.request.Request(WANDB_URL, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def safe(val, fmt=".3f"):
    if val is None:
        return "N/A"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def assess(name, run_id):
    try:
        resp = gql(run_id)
        run = resp["data"]["project"]["run"]
    except Exception as e:
        return f"**{name}** | ERROR: {e}", "ERROR"

    state = run.get("state", "unknown")

    # Parse sampledHistory — list of JSON strings, each a list of row dicts
    history = []
    for row_json in (run.get("sampledHistory") or []):
        rows = json.loads(row_json) if isinstance(row_json, str) else row_json
        history.extend(rows)

    history.sort(key=lambda r: r.get("train/global_step") or 0)

    def last(key):
        for r in reversed(history):
            if key in r and r[key] is not None:
                return r[key]
        return None

    step      = int(last("train/global_step") or 0)
    reward    = last("train/reward")
    gen_mean  = last("train/train/reward_raw/dialect_gen_mean")
    gain_mean = last("train/train/reward_raw/dialect_gain_mean")
    kl        = last("train/kl")
    grad      = last("train/grad_norm")
    length    = last("train/completions/mean_length")

    def trend(key):
        vals = [r[key] for r in history if key in r and r[key] is not None]
        if len(vals) < 4:
            return None
        n = len(vals) // 4
        return sum(vals[-n:]) / n - sum(vals[:n]) / n

    reward_trend = trend("train/reward")
    gen_trend    = trend("train/train/reward_raw/dialect_gen_mean")

    flags = []
    if state != "running":
        verdict = "STOPPED"
    elif length is not None and float(length) < 10:
        verdict = "COLLAPSED"
        flags.append("⚠ mean_length<10")
    elif kl is not None and float(kl) > 1.0:
        verdict = "UNSTABLE"
        flags.append("🔴 kl>1.0")
    elif kl is not None and float(kl) > 0.5:
        verdict = "UNSTABLE"
        flags.append("⚠ kl>0.5")
    elif grad is not None and float(grad) > 100:
        verdict = "UNSTABLE"
        flags.append("⚠ grad>100")
    elif reward is not None and float(reward) > 0 and reward_trend is not None and reward_trend > 0:
        verdict = "LEARNING" if (gen_trend is not None and gen_trend > 0) else "HEALTHY"
    elif reward is not None and float(reward) <= 0:
        verdict = "LEARNING"
    else:
        verdict = "HEALTHY"

    gain_str = (f"+{safe(gain_mean)}" if (gain_mean is not None and float(gain_mean) >= 0)
                else safe(gain_mean))

    line = (
        f"**{name}** | step {step}/5000 | reward: {safe(reward, '.2f')} | "
        f"dialect_gen: {safe(gen_mean)} | gain: {gain_str} | "
        f"kl: {safe(kl)} | grad: {safe(grad, '.1f')} | verdict: {verdict}"
    )
    if flags:
        line += "  " + " ".join(flags)
    return line, verdict


now = datetime.now(timezone.utc).strftime("%H:%M")
lines = [f"**GSPO Hourly Check** ({now} UTC)\n"]

verdicts = []
for model_name, run_id in RUNS.items():
    line, verdict = assess(model_name, run_id)
    lines.append(line)
    verdicts.append(verdict)

if any(v == "COLLAPSED" for v in verdicts):
    rec = "STOP RECOMMENDED"
elif any(v in ("UNSTABLE", "ERROR") for v in verdicts):
    rec = "MONITOR"
else:
    rec = "YES"

lines.append(f"\n**Continue running?** {rec}")
message = "\n".join(lines)

print(message)

# Post to Discord
payload = json.dumps({"content": message}).encode()
req = urllib.request.Request(DISCORD_WEBHOOK, data=payload,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        print(f"\n[Discord] {r.status} OK")
except urllib.error.HTTPError as e:
    print(f"\n[Discord] Error {e.status}: {e.read()}")
