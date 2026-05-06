from __future__ import annotations

from forseebench.generation.build_dataset import split_examples_by_movie


def test_split_examples_by_movie_has_no_leakage() -> None:
    examples = []
    for movie in ("MovieA", "MovieB", "MovieC"):
        for idx in range(2):
            examples.append({"id": f"{movie}-{idx}", "movie": movie})
    splits = split_examples_by_movie(examples, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    memberships = {}
    for split_name, rows in splits.items():
        for row in rows:
            movie = row["movie"]
            memberships.setdefault(movie, set()).add(split_name)
    assert all(len(split_names) == 1 for split_names in memberships.values())
