"""Learning Path Predictor: mock data, a small course language model, recommendations and itemsets.

Think of each student as a shopping basket of course IDs. The model reads those baskets the way a
language model reads sentences: a "word" is a course ID, a "sentence" is one student's path.
Only the standard library is used here, so every step can be checked by hand.
"""
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

START, END = "<s>", "</s>"   # markers: a path begins with <s> and ends with </s>

CATALOG = {
    "CS101": "Intro to Programming",
    "CS102": "Data Structures",
    "CS201": "Algorithms",
    "CS202": "Databases",
    "CS301": "Operating Systems",
    "MA101": "Calculus I",
    "MA102": "Linear Algebra",
    "MA201": "Probability",
    "MA202": "Statistics",
    "DS201": "Machine Learning",
    "DS202": "Data Visualization",
    "DS301": "Deep Learning",
    "WD101": "Web Fundamentals",
    "WD201": "Backend Development",
    "WD301": "Cloud Deployment",
    "HU101": "Technical Writing",
    "BA101": "Intro to Business",
    "PH101": "Physics I",
}

# The "typical" orders students follow. The generator walks along one of these and adds noise.
TRACKS = {
    "software": ["CS101", "CS102", "CS201", "CS202", "CS301"],
    "data_science": ["CS101", "MA101", "MA102", "MA201", "DS201", "DS301"],
    "web": ["CS101", "WD101", "WD201", "CS202", "WD301"],
    "analytics": ["MA101", "MA201", "MA202", "DS202", "DS201"],
}
ELECTIVES = ["HU101", "BA101", "PH101"]


# --- Milestone 1: Generate mock student course completion data as sequences of course IDs.
def generate_students(n_students: int = 500, seed: int = 0) -> list[list[str]]:
    """Make `n_students` mock paths. Each one follows a track for a random number of courses, then
    maybe swaps two neighbours, skips a course, or adds an elective. Same seed, same data."""
    rng = random.Random(seed)
    track_names = list(TRACKS)
    students = []
    for _ in range(n_students):
        track = TRACKS[rng.choice(track_names)]
        path = list(track[: rng.randint(2, len(track))])   # a student has finished the first few
        if len(path) > 3 and rng.random() < 0.15:           # took two neighbours in the other order
            i = rng.randrange(1, len(path) - 1)
            path[i], path[i + 1] = path[i + 1], path[i]
        if len(path) > 2 and rng.random() < 0.10:           # skipped one course
            del path[rng.randrange(1, len(path))]
        if rng.random() < 0.15:                             # took an unrelated elective
            path.insert(rng.randint(1, len(path)), rng.choice(ELECTIVES))
        students.append(path)
    return students


# --- Milestone 2: Pretrain a small language model on these course ID sequences.
@dataclass
class CourseModel:
    n: int                      # n-gram size: 2 = looks at 1 previous course, 3 = looks at 2
    k: float                    # add-k smoothing
    counts: dict                # context tuple -> Counter of the course that followed it
    vocab: list                 # every course seen in training, plus the END marker
    avg_position: dict          # course -> average place in a student's path (0 = first)
    n_sequences: int            # how many student paths it learned from


def train_model(sequences: list[list[str]], n: int = 3, k: float = 0.01) -> CourseModel:
    """Count, for every run of courses, what came next. A context of every length from 0 to n-1 is
    counted, so the model can back off to a shorter context when it has not seen the longer one."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if k <= 0:
        raise ValueError("k must be above 0")
    counts: dict[tuple, Counter] = defaultdict(Counter)
    positions: dict[str, list[int]] = defaultdict(list)
    for seq in sequences:
        for i, course in enumerate(seq):
            positions[course].append(i)
        tokens = [START] * (n - 1) + list(seq) + [END]
        for i in range(n - 1, len(tokens)):
            for m in range(n):                              # context lengths 0, 1, ... n-1
                counts[tuple(tokens[i - m:i])][tokens[i]] += 1
    courses = sorted(positions)
    return CourseModel(
        n=n, k=k, counts=dict(counts), vocab=courses + [END],
        avg_position={c: sum(p) / len(p) for c, p in positions.items()},
        n_sequences=len(sequences),
    )


# --- Milestone 3: Implement a function to predict the next likely course given a partial sequence.
def next_probabilities(model: CourseModel, history: list[str]) -> dict[str, float]:
    """P(next | last n-1 courses) for every token in the vocabulary, with add-k smoothing:
    (count(context, w) + k) / (count(context) + k * V). If the context was never seen, back off to
    a shorter one. The probabilities add up to 1."""
    padded = [START] * (model.n - 1) + list(history)
    context = tuple(padded[-(model.n - 1):])
    while context and context not in model.counts:
        context = context[1:]
    seen = model.counts.get(context, Counter())
    denominator = sum(seen.values()) + model.k * len(model.vocab)
    return {w: (seen[w] + model.k) / denominator for w in model.vocab}


def recommend_next(model: CourseModel, completed: list[str], top_n: int = 5,
                   reorder: bool = True) -> list[tuple[str, float]]:
    """The `top_n` most likely next courses as (course_id, probability), best first.

    Courses already completed and the END marker are left out and the rest are scaled to add up to 1.
    Students tick courses in any order, so with `reorder` the courses are first put in the order
    students usually take them (their average position in the training data). Unknown IDs are ignored.
    """
    known = [c for c in dict.fromkeys(completed) if c in model.avg_position]   # no repeats
    history = sorted(known, key=lambda c: (model.avg_position[c], c)) if reorder else known
    probs = next_probabilities(model, history)
    left = {c: p for c, p in probs.items() if c != END and c not in known}
    total = sum(left.values())
    if total == 0:
        return []
    ranked = sorted(left.items(), key=lambda item: (-item[1], item[0]))
    return [(c, p / total) for c, p in ranked[:top_n]]


# --- Milestone 4: Create a Streamlit interface for students to input their completed courses and view recommendations.
def recommendation_table(model: CourseModel, completed: list[str], top_n: int = 5) -> list[dict]:
    """Rows the Streamlit page can show directly: rank, course ID, title and probability."""
    return [
        {"rank": i, "course": c, "title": CATALOG.get(c, c), "probability": p}
        for i, (c, p) in enumerate(recommend_next(model, completed, top_n), start=1)
    ]


# --- Milestone 5: Analyze which course "itemsets" the model has learned.
def learned_itemsets(model: CourseModel, size: int = 2, top_n: int = 10) -> list[tuple[tuple, int]]:
    """The `top_n` runs of `size` consecutive courses the model saw most often, as (itemset, count).
    Start and end markers are not courses, so runs that include them are skipped."""
    if not 1 <= size <= model.n:
        raise ValueError(f"size must be between 1 and {model.n}")
    found = []
    for context, followers in model.counts.items():
        if len(context) != size - 1:
            continue
        for course, count in followers.items():
            itemset = context + (course,)
            if START not in itemset and END not in itemset:
                found.append((itemset, count))
    found.sort(key=lambda item: (-item[1], item[0]))
    return found[:top_n]
