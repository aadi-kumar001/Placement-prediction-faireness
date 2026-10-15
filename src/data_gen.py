"""Synthetic campus-placement data with a *deliberately injected* historical bias.

Why synthetic? Real placement data is private. Synthetic data lets us KNOW the ground truth,
so we can prove that our fairness tooling detects the bias we planted.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42


def generate(n: int = 4000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], n, p=[0.58, 0.42])
    region = rng.choice(["Urban", "Rural"], n, p=[0.65, 0.35])
    income = rng.choice(["Low", "Mid", "High"], n, p=[0.30, 0.50, 0.20])
    tier = rng.choice([1, 2, 3], n, p=[0.20, 0.45, 0.35])
    branch = rng.choice(["CSE", "ECE", "Mech", "Civil", "IT"], n, p=[0.30, 0.20, 0.18, 0.12, 0.20])

    cgpa = np.clip(rng.normal(7.3, 1.0, n), 4.5, 10)
    aptitude = np.clip(rng.normal(60 + 3 * (cgpa - 7), 12, n), 10, 100)
    communication = np.clip(rng.normal(6 + (region == "Urban") * 0.8 + (income == "High") * 0.5, 1.6, n), 1, 10)
    internships = np.clip(rng.poisson(0.8 + 0.5 * (tier == 1) + 0.4 * (cgpa > 8), n), 0, 5)
    projects = np.clip(rng.poisson(1.8 + 0.4 * (branch == "CSE"), n), 0, 8)
    certs = np.clip(rng.poisson(1.5, n), 0, 8)
    backlogs = np.clip(rng.poisson(np.where(cgpa < 6.5, 1.2, 0.2), n), 0, 6)

    # Merit signal (what SHOULD drive hiring)
    merit = (0.9 * (cgpa - 7) + 0.03 * (aptitude - 60) + 0.25 * (communication - 6)
             + 0.45 * internships + 0.20 * projects + 0.10 * certs - 0.6 * backlogs
             + 0.4 * (branch == "CSE") + 0.25 * (branch == "IT"))

    # Historical bias (what should NOT drive hiring) -- planted on purpose
    bias = (0.55 * (gender == "Male") + 0.35 * (region == "Urban")
            + 0.20 * (income == "High") - 0.10 * (income == "Low"))

    logit = -1.0 + 0.9 * merit + bias - 0.35 * (tier - 2)
    prob = 1 / (1 + np.exp(-logit))
    placed = (rng.random(n) < prob).astype(int)
    salary = np.where(placed == 1, np.round(np.clip(rng.normal(5.5 + 1.2 * merit, 1.5), 3, 30), 2), 0.0)

    return pd.DataFrame(dict(
        student_id=np.arange(1, n + 1), gender=gender, region=region, family_income=income,
        college_tier=tier, branch=branch, cgpa=cgpa.round(2), aptitude_score=aptitude.round(1),
        communication=communication.round(1), internships=internships, projects=projects,
        certifications=certs, backlogs=backlogs, placed=placed, salary_lpa=salary))


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/placement_data.csv", index=False)
    print(df.shape, "placement rate:", df.placed.mean().round(3))
