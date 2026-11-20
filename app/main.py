"""Streamlit entry point: `streamlit run app/main.py`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `app.core` imports work

import pandas as pd
import streamlit as st

from app.core import (CATALOG, generate_students, learned_itemsets, recommendation_table,
                      train_model)

st.set_page_config(page_title="Learning Path Predictor")
st.title("Learning Path Predictor")
st.write("For students, recommends next courses or study materials based on their completed courses, optimizing learning paths and suggesting relevant subjects.")


@st.cache_resource
def load_model():
    """Make the mock students and train the model once, then reuse it on every click."""
    return train_model(generate_students(n_students=500, seed=42), n=3, k=0.01)


model = load_model()
label = lambda course: f"{course} - {CATALOG[course]}"

st.header("Your next courses")
completed = st.multiselect("Courses you have completed", options=list(CATALOG), format_func=label)
top_n = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)

if completed:
    rows = recommendation_table(model, completed, top_n)
else:
    st.caption("Nothing selected yet, so these are the courses most students start with.")
    rows = recommendation_table(model, [], top_n)

if rows:
    table = pd.DataFrame(rows).rename(columns={"rank": "#", "course": "Course", "title": "Title",
                                               "probability": "Chance"})
    table["Chance"] = table["Chance"] * 100                       # show as a percentage
    st.dataframe(table, hide_index=True, column_config={
        "Chance": st.column_config.ProgressColumn("Chance", format="%.0f%%", min_value=0, max_value=100)})
    st.bar_chart(table.set_index("Course")["Chance"])
else:
    st.success("You have completed every course in the catalog.")

st.header("What the model has learned")
size = st.radio("Course pattern length", options=[2, 3], format_func=lambda s: f"{s} courses in a row",
                horizontal=True)
itemsets = learned_itemsets(model, size=size, top_n=10)
st.dataframe(pd.DataFrame([
    {"Pattern": " → ".join(items), "Students": count, "Share": 100 * count / model.n_sequences}
    for items, count in itemsets
]), hide_index=True, column_config={"Share": st.column_config.NumberColumn("Share", format="%.1f%%")})