"""Generative-AI layer. Uses the Claude API when ANTHROPIC_API_KEY is set; otherwise falls back to
deterministic templates so the whole project still runs offline (great for demos & tests)."""
import json, os
import numpy as np, pandas as pd

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
NUM_LABELS = {"cgpa": "CGPA", "aptitude_score": "aptitude score", "communication": "communication rating",
              "internships": "internships", "projects": "projects", "certifications": "certifications",
              "backlogs": "backlogs", "college_tier": "college tier"}


def _llm(system: str, user: str, max_tokens: int = 500):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        r = anthropic.Anthropic(api_key=key).messages.create(
            model=MODEL, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        return r.content[0].text.strip()
    except Exception as e:          # never crash the dashboard because of the LLM
        return f"(LLM unavailable: {e})"


# ---------- model-grounded facts the LLM is allowed to use ----------
def local_drivers(pipe, features, row: pd.DataFrame, reference: pd.DataFrame, k=4):
    """Counterfactual attribution: how much does P(placed) change if a feature is set to the cohort median?"""
    base = pipe.predict_proba(row[features])[0, 1]
    out = []
    for f in [c for c in features if c in NUM_LABELS]:
        r2 = row.copy(); r2[f] = reference[f].median()
        out.append((f, base - pipe.predict_proba(r2[features])[0, 1]))
    out.sort(key=lambda t: -abs(t[1]))
    return base, out[:k]


def what_if(pipe, features, row: pd.DataFrame):
    """Actionable, non-protected levers a student can actually change."""
    base = pipe.predict_proba(row[features])[0, 1]
    levers = {"internships": +1, "certifications": +1, "projects": +1, "backlogs": -1}
    res = []
    for f, d in levers.items():
        r2 = row.copy(); r2[f] = max(0, r2[f].iloc[0] + d)
        res.append((f, d, pipe.predict_proba(r2[features])[0, 1] - base))
    return sorted(res, key=lambda t: -t[2])


# ---------- GenAI features ----------
def explain_student(prob, drivers, levers, protected_used=False) -> str:
    facts = dict(placement_probability=round(float(prob), 3),
                 drivers=[(NUM_LABELS[f], round(float(d), 3)) for f, d in drivers],
                 what_if=[(NUM_LABELS[f], d, round(float(g), 3)) for f, d, g in levers],
                 protected_attributes_used=protected_used)
    system = ("You are a supportive campus placement mentor. Use ONLY the facts provided. Never mention or infer "
              "gender, region or income. 120 words max, plain language, end with 2 concrete next steps.")
    llm = _llm(system, json.dumps(facts))
    if llm:
        return llm
    best = levers[0]
    pos = [NUM_LABELS[f] for f, d in drivers if d > 0][:2]; neg = [NUM_LABELS[f] for f, d in drivers if d < 0][:2]
    return (f"Predicted placement chance: **{prob:.0%}**.\n\n"
            f"Strengths: {', '.join(pos) or 'none stand out yet'}. "
            f"Held back by: {', '.join(neg) or 'nothing major'}.\n\n"
            f"Biggest realistic lever: {'add' if best[1] > 0 else 'clear'} 1 {NUM_LABELS[best[0]]} "
            f"(≈ {best[2]:+.1%} chance). Protected attributes were not used in this prediction.")


def fairness_narrative(table: pd.DataFrame, metrics: dict, audience="placement officer") -> str:
    facts = dict(selection_rates={k: round(float(v), 3) for k, v in table.selection_rate.items()},
                 metrics={k: round(float(v), 3) for k, v in metrics.items() if k != "table"})
    system = (f"You write fairness audit summaries for a {audience}. Use ONLY the numbers given. Explain what the "
              "metrics mean, state pass/fail against the 4/5ths (0.80) disparate-impact rule, and give 3 recommendations. "
              "Under 180 words. Do not claim causes you cannot know.")
    llm = _llm(system, json.dumps(facts), 600)
    if llm:
        return llm
    di = metrics["disparate_impact"]; lo = table.selection_rate.idxmin(); hi = table.selection_rate.idxmax()
    verdict = "PASSES" if di >= 0.8 else "FAILS"
    return (f"**Audit summary.** '{lo}' students are selected at {table.selection_rate.min():.0%} vs "
            f"{table.selection_rate.max():.0%} for '{hi}'. Disparate impact = {di:.2f}, which {verdict} the 0.80 rule; "
            f"demographic-parity gap = {metrics['demographic_parity_diff']:.2f}.\n\n"
            "Recommendations: (1) review features that proxy protected attributes, (2) apply reweighing or "
            "threshold adjustment and re-audit, (3) have a human review borderline rejections.")


def synthetic_profiles_prompt(n=5) -> str:
    return (f"Generate {n} realistic but fictional engineering-student profiles as a JSON list with keys: cgpa (4.5-10), "
            "aptitude_score (10-100), communication (1-10), internships (0-5), projects (0-8), certifications (0-8), "
            "backlogs (0-6), branch (CSE/ECE/Mech/Civil/IT), college_tier (1-3). Cover weak, average and strong students. "
            "Return ONLY JSON.")


def generate_synthetic_profiles(n=5):
    txt = _llm("You produce only valid JSON.", synthetic_profiles_prompt(n), 900)
    if not txt:
        return None
    try:
        return pd.DataFrame(json.loads(txt.replace("```json", "").replace("```", "")))
    except Exception:
        return None


def chat_with_data(question: str, summary: dict) -> str:
    system = ("You are a data analyst for a placement cell. Answer ONLY from the JSON summary. "
              "If the answer is not in it, say so. Be brief.")
    return _llm(system, f"SUMMARY:\n{json.dumps(summary)}\n\nQUESTION: {question}") or \
        "Set ANTHROPIC_API_KEY to enable the data-chat assistant."
