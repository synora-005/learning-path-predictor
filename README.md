# Learning Path Predictor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/hskvuwr5zn/badge.svg)](https://ask.thili.ai/b/hskvuwr5zn)

The Learning Path Predictor is an EdTech project designed to help students navigate their academic journey. It
treats each student's completed courses like a shopping basket, learns which courses tend to follow which, and
recommends the most relevant next courses.

This version is a small n-gram language model over course IDs, written from scratch in plain Python (standard
library only). It learns from mock students generated in the app.

## What you are building

**Goal.** You will have built a web application that recommends the next likely courses to students based on their
completed coursework, optimizing their learning paths.

**Done when.** The project is done when the application successfully displays top N relevant course recommendations
after a user inputs their completed courses, demonstrating the model's ability to suggest logical next steps.

## Milestones

- [x] **1. Generate mock student course completion data as sequences of course IDs.**
      `generate_students` follows four typical tracks (software, data science, web, analytics) and adds noise:
      swapped neighbours, skipped courses and electives. The same seed gives the same data.
- [x] **2. Pretrain a small language model on these course ID sequences.**
      `train_model` counts what follows every context of 0 to n-1 courses, with `<s>` and `</s>` start/end markers.
- [x] **3. Implement a function to predict the next likely course given a partial sequence.**
      `next_probabilities` gives P(next | last n-1 courses) with add-k smoothing and backoff to a shorter context;
      `recommend_next` removes completed courses and returns the top N.
- [x] **4. Create a Streamlit interface for students to input their completed courses and view recommendations.**
      `app/main.py` with `recommendation_table` from `app/core.py`.
- [x] **5. Analyze which course "itemsets" the model has learned.**
      `learned_itemsets` lists the most common runs of 1, 2 or 3 courses in a row.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional: this model does not use any of the settings
streamlit run app/main.py
pytest
```

On Windows (PowerShell): `python -m venv .venv`, `.venv\Scripts\activate`, `copy .env.example .env`.

## Using the app

1. Pick the courses you have completed (in any order). With nothing picked you see the courses most students start with.
2. Choose how many recommendations you want. Each shows its chance as a percentage.
3. Scroll down to **What the model has learned** to see the most common 2-course and 3-course patterns.

## How it works

1. **Mock data.** Each student follows a track for a random number of courses, with a little noise.
2. **Count.** For every run of up to `n` courses (default 3), count how often each course came next.
3. **Smooth.** `(count + k) / (count(context) + k * V)` so a course never seen after a context still has a small chance.
4. **Back off.** If the last two courses were never seen together, use only the last one, then no context at all.
5. **Recommend.** Put the chosen courses in the usual order (their average position in the data), score every
   course, drop the ones already completed and the end marker, scale the rest to add up to 1, and show the top N.
6. **Itemsets.** Read the counts back out: the runs of courses with the highest counts are what the model learned.

## Tests

```bash
pytest
```

Each test uses a tiny made-up set of paths so the expected numbers can be checked by hand.

## Settings and secrets

`.env` is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever reaches GitHub, treat it as
leaked and make a new one. This project does not call a hosted model, so no key is needed.

## Limits and ideas

- The data is mock, so the recommendations only show the tracks the generator was told about, not real student behaviour.
- A trigram model sees only the last two courses; a student's earlier courses are ignored once the path is long.
- Ideas to try: let students pick a learning goal, recommend study materials as well as courses, chart the learned
  patterns, or add prerequisites so a course is only suggested when its prerequisites are done.