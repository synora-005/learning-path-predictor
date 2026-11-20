"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

Each one uses a tiny made-up set of paths so the expected answers can be checked by hand.
"""
import pytest

from app.core import (CATALOG, END, START, generate_students, learned_itemsets, next_probabilities,
                      recommend_next, recommendation_table, train_model)

TINY = [["A", "B"], ["A", "C"], ["A", "B"]]


# Milestone 1: Generate mock student course completion data as sequences of course IDs.
def test_milestone_1():
    students = generate_students(100, seed=1)
    assert len(students) == 100
    for path in students:
        assert len(path) >= 2                       # every student has finished something
        assert len(set(path)) == len(path)          # nobody takes the same course twice
        assert all(c in CATALOG for c in path)      # only real course IDs
    assert students == generate_students(100, seed=1)      # same seed, same data
    assert students != generate_students(100, seed=2)


# Milestone 2: Pretrain a small language model on these course ID sequences.
def test_milestone_2():
    model = train_model(TINY, n=2, k=0.5)
    assert model.counts[("A",)]["B"] == 2           # B followed A twice
    assert model.counts[(START,)]["A"] == 3         # all three paths began with A
    assert model.counts[("B",)][END] == 2           # and two paths ended after B
    assert model.vocab == ["A", "B", "C", END]
    # P(next | A): (count + k) / (3 + k*V) with k=0.5 and V=4 is (count + 0.5) / 5
    probs = next_probabilities(model, ["A"])
    assert probs["B"] == pytest.approx(0.5)
    assert probs["C"] == pytest.approx(0.3)
    assert probs["A"] == pytest.approx(0.1)         # smoothing: never impossible
    assert sum(probs.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        train_model(TINY, n=1)


# Milestone 3: Implement a function to predict the next likely course given a partial sequence.
def test_milestone_3():
    model = train_model([["A", "B", "C"]] * 3 + [["A", "B", "D"]], n=2, k=0.001)
    assert recommend_next(model, ["A"])[0][0] == "B"
    top = recommend_next(model, ["A", "B"], top_n=2)
    assert [c for c, _ in top] == ["C", "D"]                       # 3 students vs 1
    everything = recommend_next(model, ["A"], top_n=10)
    assert "A" not in [c for c, _ in everything]                   # already completed
    assert sum(p for _, p in everything) == pytest.approx(1.0)
    assert recommend_next(model, ["B", "A"]) == recommend_next(model, ["A", "B"])   # order of ticking
    assert recommend_next(model, [])[0][0] == "A"                  # new student: where people start
    assert recommend_next(model, ["A", "B", "C", "D"]) == []       # nothing left to suggest
    assert recommend_next(model, ["A", "NOPE"]) == recommend_next(model, ["A"])     # unknown ID ignored
    # an unseen longer context backs off to a shorter one
    trigram = train_model([["A", "B", "C"]], n=3, k=0.001)
    probs = next_probabilities(trigram, ["Z", "B"])
    assert max(probs, key=probs.get) == "C"


# Milestone 4: Create a Streamlit interface for students to input their completed courses and view recommendations.
def test_milestone_4():
    model = train_model(generate_students(500, seed=42))
    rows = recommendation_table(model, ["CS101", "CS102"], top_n=3)
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["course"] == "CS201"             # Intro -> Data Structures -> Algorithms
    assert rows[0]["title"] == CATALOG["CS201"]
    assert rows[0]["probability"] > rows[1]["probability"] > 0


# Milestone 5: Analyze which course "itemsets" the model has learned.
def test_milestone_5():
    model = train_model(TINY, n=2, k=0.5)
    assert learned_itemsets(model, size=2) == [(("A", "B"), 2), (("A", "C"), 1)]   # no <s> or </s>
    assert learned_itemsets(model, size=1) == [(("A",), 3), (("B",), 2), (("C",), 1)]
    assert learned_itemsets(model, size=2, top_n=1) == [(("A", "B"), 2)]
    with pytest.raises(ValueError):
        learned_itemsets(model, size=3)             # the model only looks 1 course back
    real = train_model(generate_students(500, seed=42))
    pairs = [items for items, _ in learned_itemsets(real, size=2, top_n=10)]
    assert ("CS101", "CS102") in pairs