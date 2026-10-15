"""End-to-end: data -> features -> models -> fairness audit -> mitigation -> figures + artifacts."""
import json, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from data_gen import generate
from fairness import (fairness_report, reweighing_weights, group_thresholds, apply_thresholds)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
FIG = "reports/figures/"
PROTECTED = ["gender", "region", "family_income"]
NUM = ["cgpa", "aptitude_score", "communication", "internships", "projects",
       "certifications", "backlogs", "college_tier"]
CAT_NEUTRAL = ["branch"]


def save(name):
    plt.tight_layout(); plt.savefig(FIG + name, dpi=150, bbox_inches="tight"); plt.close()


def build(model, cat):
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("clf", model)])


# ---------- 1. data ----------
df = generate(); df.to_csv("data/placement_data.csv", index=False)
df["gender_region"] = df.gender + "_" + df.region
train, test = train_test_split(df, test_size=0.3, stratify=df.placed, random_state=42)
ytr, yte = train.placed.values, test.placed.values

# ---------- 2. EDA figures ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["gender", "region", "family_income"]):
    order = ["Low", "Mid", "High"] if col == "family_income" else None
    r = df.groupby(col).placed.mean().reindex(order) if order else df.groupby(col).placed.mean()
    sns.barplot(x=r.index, y=r.values, ax=a)
    a.set_title(f"Placement rate by {col}"); a.set_ylim(0, 1); a.set_ylabel("placement rate")
    for i, v in enumerate(r.values): a.text(i, v + 0.015, f"{v:.1%}", ha="center")
save("01_placement_rate_by_group.png")

plt.figure(figsize=(8, 6.5))
sns.heatmap(df[NUM + ["placed"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature correlation matrix"); save("02_correlation_heatmap.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=df, x="cgpa", hue="placed", kde=True, ax=ax[0]); ax[0].set_title("CGPA distribution by outcome")
sns.boxplot(data=df, x="internships", y="aptitude_score", hue="placed", ax=ax[1]); ax[1].set_title("Aptitude vs internships")
save("03_eda_distributions.png")

# ---------- 3. model comparison (all features, incl. protected) ----------
ALL_CAT = CAT_NEUTRAL + PROTECTED
models = {"Logistic Regression": LogisticRegression(max_iter=1000),
          "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, random_state=42, n_jobs=-1),
          "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
metrics, plt_roc = {}, plt.figure(figsize=(6.5, 5))
for name, m in models.items():
    p = build(m, ALL_CAT).fit(train[NUM + ALL_CAT], ytr)
    s = p.predict_proba(test[NUM + ALL_CAT])[:, 1]
    metrics[name] = dict(accuracy=accuracy_score(yte, s >= .5), f1=f1_score(yte, s >= .5), auc=roc_auc_score(yte, s))
    fpr, tpr, _ = roc_curve(yte, s); plt.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves - model comparison"); plt.legend(); save("04_roc_curves.png")

# ---------- 4. the four fairness scenarios (Gradient Boosting) ----------
gb = lambda: GradientBoostingClassifier(random_state=42)
NEUTRAL_CAT = CAT_NEUTRAL
scen = {}

# A: Biased baseline - sees protected attributes
mA = build(gb(), ALL_CAT).fit(train[NUM + ALL_CAT], ytr)
sA = mA.predict_proba(test[NUM + ALL_CAT])[:, 1]; scen["A. Baseline (uses protected attrs)"] = (sA, (sA >= .5).astype(int))

# B: Fairness through unawareness - drop protected columns
mB = build(gb(), NEUTRAL_CAT).fit(train[NUM + NEUTRAL_CAT], ytr)
sB = mB.predict_proba(test[NUM + NEUTRAL_CAT])[:, 1]; scen["B. Unaware (drop protected)"] = (sB, (sB >= .5).astype(int))

# C: Reweighing on intersectional group (gender x region) - pre-processing
w = reweighing_weights(ytr, train.gender_region.values)
mC = build(gb(), NEUTRAL_CAT); mC.fit(train[NUM + NEUTRAL_CAT], ytr, clf__sample_weight=w)
sC = mC.predict_proba(test[NUM + NEUTRAL_CAT])[:, 1]; scen["C. Reweighing (pre-process)"] = (sC, (sC >= .5).astype(int))

# D: Group thresholds on model B scores - post-processing (equalise selection rate across gender x region)
rate = (sB >= .5).mean()
th = group_thresholds(sB, test.gender_region.values, rate)
yD = apply_thresholds(sB, test.gender_region.values, th); scen["D. Group thresholds (post-process)"] = (sB, yD)

rows = []
for name, (s, yp) in scen.items():
    row = dict(scenario=name, accuracy=accuracy_score(yte, yp), f1=f1_score(yte, yp), auc=roc_auc_score(yte, s))
    for attr in ["gender", "region", "gender_region"]:
        fr = fairness_report(yte, yp, test[attr].values)
        row.update({f"{attr}_dpd": fr["demographic_parity_diff"], f"{attr}_di": fr["disparate_impact"],
                    f"{attr}_eod": fr["equal_opportunity_diff"]})
    rows.append(row)
res = pd.DataFrame(rows); res.to_csv("reports/scenario_results.csv", index=False)
print(res.round(3).to_string())

# ---------- 5. fairness figures ----------
short = ["A Baseline", "B Unaware", "C Reweigh", "D GroupThr"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a, (col, ttl) in zip(ax, [("gender_region_dpd", "Demographic parity difference (lower=fairer)"),
                              ("gender_region_eod", "Equal opportunity difference (lower=fairer)"),
                              ("gender_region_di", "Disparate impact ratio (>=0.80 passes)")]):
    sns.barplot(x=short, y=res[col], ax=a)
    a.set_title(ttl, fontsize=10)
    for i, v in enumerate(res[col]): a.text(i, v + 0.005, f"{v:.2f}", ha="center")
    if col.endswith("_di"): a.axhline(0.8, color="red", ls="--"); a.set_ylim(0, 1.05)
save("05_fairness_metrics_comparison.png")

# group selection rates (A vs D)
plt.figure(figsize=(9, 4.5))
tab = pd.DataFrame({n.split(".")[0] + " " + n.split(". ")[1].split(" (")[0]:
                    pd.Series(yp).groupby(test.gender_region.values).mean() for n, (s, yp) in scen.items()})
tab.plot(kind="bar", ax=plt.gca(), width=.8); plt.ylabel("selection rate"); plt.xticks(rotation=0)
plt.title("Predicted selection rate by intersectional group"); plt.legend(fontsize=8); save("06_selection_rate_by_group.png")

# accuracy vs fairness trade-off
plt.figure(figsize=(6.5, 5))
plt.scatter(res.gender_region_dpd, res.accuracy, s=140, c=sns.color_palette("deep", 4))
for i, n in enumerate(short): plt.annotate(n, (res.gender_region_dpd[i] + .003, res.accuracy[i] + .001))
plt.xlabel("Demographic parity difference (gender x region)"); plt.ylabel("Accuracy")
plt.title("Accuracy vs fairness trade-off"); save("07_accuracy_fairness_tradeoff.png")

# confusion matrices A vs C
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, (n, i) in zip(ax, [("A Baseline", 0), ("C Reweighing", 2)]):
    ConfusionMatrixDisplay(confusion_matrix(yte, list(scen.values())[i][1]),
                           display_labels=["Not placed", "Placed"]).plot(ax=a, colorbar=False, cmap="Blues"); a.set_title(n)
save("08_confusion_matrices.png")

# permutation importance (baseline, shows protected attrs leak into model)
pi = permutation_importance(mA, test[NUM + ALL_CAT], yte, n_repeats=8, random_state=42, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=NUM + ALL_CAT).sort_values()
plt.figure(figsize=(7.5, 5.5))
colors = ["#d62728" if c in PROTECTED else "#4c72b0" for c in imp.index]
plt.barh(imp.index, imp.values, color=colors); plt.xlabel("Drop in AUC when shuffled")
plt.title("Permutation importance - red = protected attributes"); save("09_feature_importance.png")

# proxy leakage: can protected attrs be predicted from "neutral" features?
proxy = {}
for attr in ["gender", "region", "family_income"]:
    pm = build(LogisticRegression(max_iter=1000), NEUTRAL_CAT).fit(train[NUM + NEUTRAL_CAT], train[attr])
    proxy[attr] = float(pm.score(test[NUM + NEUTRAL_CAT], test[attr]))
maj = {a: float(test[a].value_counts(normalize=True).iloc[0]) for a in proxy}
plt.figure(figsize=(6.5, 4)); x = np.arange(3)
plt.bar(x - .2, [maj[a] for a in proxy], .4, label="Majority-class guess"); plt.bar(x + .2, list(proxy.values()), .4, label="Predicted from 'neutral' features")
plt.xticks(x, list(proxy)); plt.ylim(0, 1); plt.legend(); plt.title("Proxy leakage test"); save("10_proxy_leakage.png")

# ---------- 6. architecture diagram ----------
fig, ax = plt.subplots(figsize=(14, 4.6)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 4.6)
boxes = [("1 Data\nSynthetic / Real CSV", .2, "#cfe2ff"), ("2 Features &\nPreprocessing", 2.5, "#cfe2ff"),
         ("3 ML Models\nLR / RF / GB", 4.8, "#d1e7dd"), ("4 Fairness Audit\nDPD / EOD / DI", 7.1, "#fff3cd"),
         ("5 Mitigation\nReweigh / Thresholds", 9.4, "#fff3cd"), ("6 Dashboard\nStreamlit", 11.7, "#f8d7da")]
for t, x0, c in boxes:
    ax.add_patch(FancyBboxPatch((x0, 2.2), 2.0, 1.6, boxstyle="round,pad=.05", fc=c, ec="k")); ax.text(x0 + 1, 3, t, ha="center", va="center", fontsize=10)
for x0 in [0.2, 2.5, 4.8, 7.1, 9.4]: ax.annotate("", (x0 + 2.5, 3), (x0 + 2.0, 3), arrowprops=dict(arrowstyle="-|>", lw=2))
ax.add_patch(FancyBboxPatch((4.8, .3), 6.6, 1.1, boxstyle="round,pad=.05", fc="#e2d9f3", ec="k"))
ax.text(8.1, .85, "Generative AI layer (Claude API): natural-language explanations, fairness-report narration,\ncounterfactual coaching, synthetic profile generation", ha="center", va="center", fontsize=9.5)
ax.annotate("", (8.1, 1.4), (8.1, 2.2), arrowprops=dict(arrowstyle="<->", lw=2, color="purple"))
ax.set_title("System architecture", fontsize=13); save("00_architecture.png")

# ---------- 7. dashboard preview (static composite) ----------
fig = plt.figure(figsize=(14, 7)); gs = fig.add_gridspec(2, 3)
fig.suptitle("Placement Prediction & Fairness Dashboard - static preview", fontsize=14, fontweight="bold")
a = fig.add_subplot(gs[0, 0]); a.axis("off")
for i, (k, v) in enumerate([("Accuracy", res.accuracy[2]), ("AUC", res.auc[2]), ("Disparate impact", res.gender_region_di[2]), ("Parity gap", res.gender_region_dpd[2])]):
    a.text(.05 + (i % 2) * .5, .8 - (i // 2) * .45, f"{k}\n{v:.3f}", fontsize=16, va="top", bbox=dict(boxstyle="round", fc="#eef")); 
a.set_title("KPI cards (Reweighing model)")
a = fig.add_subplot(gs[0, 1]); sns.barplot(x=short, y=res.gender_region_dpd, ax=a); a.set_title("Parity gap by scenario")
a = fig.add_subplot(gs[0, 2]); tab.plot(kind="bar", ax=a, legend=False); a.set_title("Selection rate by group")
a = fig.add_subplot(gs[1, 0]); a.barh(imp.index[-6:], imp.values[-6:]); a.set_title("Top drivers")
a = fig.add_subplot(gs[1, 1:]); a.axis("off")
a.text(0, .9, "AI explanation (GenAI tab)", fontweight="bold", fontsize=12)
a.text(0, .75, "This student has a ~78% predicted chance of placement.\nStrengths: 2 internships, CGPA 8.1, strong aptitude.\nTo improve: add one certification and reduce backlogs.\nGender/region were NOT used in this prediction.", fontsize=11, va="top",
       bbox=dict(boxstyle="round", fc="#f6f0ff"))
save("11_dashboard_preview.png")

# ---------- 8. persist artifacts ----------
final = mC   # reweighed model = recommended
joblib.dump(dict(model=final, features=NUM + NEUTRAL_CAT, baseline=mA, baseline_features=NUM + ALL_CAT), "models/final_model.joblib")
test.assign(score_baseline=sA, score_final=sC, pred_baseline=(sA >= .5).astype(int), pred_final=(sC >= .5).astype(int)
            ).to_csv("data/test_predictions.csv", index=False)
json.dump(dict(model_comparison=metrics, proxy_accuracy=proxy, majority=maj, group_thresholds=th), open("reports/metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=1)); print("proxy", proxy, "maj", maj)
