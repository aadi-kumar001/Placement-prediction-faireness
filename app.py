"""Streamlit dashboard:  streamlit run app.py"""
import json, sys
sys.path.insert(0, "src")
import joblib, pandas as pd, plotly.express as px, streamlit as st
from fairness import fairness_report
import genai

st.set_page_config(page_title="Placement & Fairness Dashboard", layout="wide", page_icon="🎓")
art = joblib.load("models/final_model.joblib")
df = pd.read_csv("data/test_predictions.csv")
res = pd.read_csv("reports/scenario_results.csv")
metrics = json.load(open("reports/metrics.json"))
full = pd.read_csv("data/placement_data.csv")

st.title("🎓 End-to-End Placement Prediction & Fairness Dashboard")
tabs = st.tabs(["Overview", "Predict a student", "Fairness audit", "Mitigation lab", "GenAI assistant"])

with tabs[0]:
    c = st.columns(4)
    c[0].metric("Students", f"{len(full):,}"); c[1].metric("Placement rate", f"{full.placed.mean():.1%}")
    c[2].metric("Final model AUC", f"{res.auc[2]:.3f}"); c[3].metric("Final model accuracy", f"{res.accuracy[2]:.1%}")
    col = st.selectbox("Placement rate by", ["gender", "region", "family_income", "branch", "college_tier"])
    st.plotly_chart(px.bar(full.groupby(col).placed.mean().reset_index(), x=col, y="placed", text_auto=".1%"), use_container_width=True)
    st.dataframe(pd.DataFrame(metrics["model_comparison"]).T.style.format("{:.3f}"))

with tabs[1]:
    st.subheader("What-if placement predictor")
    a, b, c3 = st.columns(3)
    cgpa = a.slider("CGPA", 4.5, 10.0, 7.5, .1); apt = a.slider("Aptitude", 10, 100, 65)
    comm = b.slider("Communication", 1.0, 10.0, 6.5, .5); intern = b.slider("Internships", 0, 5, 1)
    proj = c3.slider("Projects", 0, 8, 2); cert = c3.slider("Certifications", 0, 8, 1)
    back = a.slider("Backlogs", 0, 6, 0); tier = b.selectbox("College tier", [1, 2, 3], 1)
    branch = c3.selectbox("Branch", ["CSE", "IT", "ECE", "Mech", "Civil"])
    row = pd.DataFrame([dict(cgpa=cgpa, aptitude_score=apt, communication=comm, internships=intern, projects=proj,
                             certifications=cert, backlogs=back, college_tier=tier, branch=branch)])
    prob, drivers = genai.local_drivers(art["model"], art["features"], row, full)
    levers = genai.what_if(art["model"], art["features"], row)
    st.progress(float(prob), text=f"Placement probability: {prob:.1%}")
    st.markdown(genai.explain_student(prob, drivers, levers))
    st.plotly_chart(px.bar(pd.DataFrame(drivers, columns=["feature", "impact"]), x="impact", y="feature", orientation="h",
                           title="What moves this prediction (vs. cohort median)"), use_container_width=True)

with tabs[2]:
    st.subheader("Fairness audit on the held-out test set")
    attr = st.selectbox("Protected group", ["gender", "region", "gender_region"])
    which = st.radio("Model", ["Baseline (uses protected attrs)", "Final (reweighed, attrs removed)"], horizontal=True)
    pred = df.pred_baseline if which.startswith("Base") else df.pred_final
    fr = fairness_report(df.placed.values, pred.values, df[attr].values)
    m = st.columns(3)
    m[0].metric("Demographic parity diff", f"{fr['demographic_parity_diff']:.3f}")
    m[1].metric("Disparate impact", f"{fr['disparate_impact']:.3f}", "pass" if fr["disparate_impact"] >= .8 else "FAIL")
    m[2].metric("Equal opportunity diff", f"{fr['equal_opportunity_diff']:.3f}")
    st.dataframe(fr["table"].style.format({c: "{:.3f}" for c in fr["table"].columns if c != "n"}))
    st.plotly_chart(px.bar(fr["table"].reset_index(), x="group", y=["selection_rate", "tpr"], barmode="group"), use_container_width=True)
    st.markdown(genai.fairness_narrative(fr["table"], fr))

with tabs[3]:
    st.subheader("Compare mitigation strategies")
    st.dataframe(res.set_index("scenario")[["accuracy", "f1", "auc", "gender_region_dpd", "gender_region_di", "gender_region_eod"]].style.format("{:.3f}"))
    st.plotly_chart(px.scatter(res, x="gender_region_dpd", y="accuracy", text="scenario", size_max=18,
                               title="Accuracy vs fairness trade-off"), use_container_width=True)

with tabs[4]:
    st.subheader("Ask questions about the results")
    q = st.text_input("e.g. Which group has the lowest selection rate and what should we do?")
    if q:
        summ = dict(scenarios=res.round(3).to_dict("records"), placement_rate_by_gender=full.groupby("gender").placed.mean().round(3).to_dict(),
                    placement_rate_by_region=full.groupby("region").placed.mean().round(3).to_dict())
        st.write(genai.chat_with_data(q, summ))
    if st.button("Generate synthetic student profiles with Claude"):
        out = genai.generate_synthetic_profiles(5)
        st.dataframe(out) if out is not None else st.info("Set ANTHROPIC_API_KEY to enable.")
