"""Recompute all nine scores from the published 900 episode records."""
import json, math, pathlib, statistics
root = pathlib.Path(__file__).resolve().parents[1]
data = root / "docs/results"
summary = json.loads((data / "formal-300k.json").read_text())
episodes = json.loads((data / "episodes-300k.json").read_text())
assert len(episodes) == 900
for run in summary["runs"]:
    rows = [e for e in episodes if (e["method"], e["training_seed"]) == (run["method"], run["seed"])]
    assert len(rows) == 100
    assert len({e["episode"] for e in rows}) == 100
    keys = sorted(rows[0]["achievements"])
    assert len(keys) == 22
    rates = {k: sum(e["achievements"][k] > 0 for e in rows) for k in keys}
    assert rates == run["achievement_success_pct"]
    score = math.expm1(statistics.mean(math.log1p(v) for v in rates.values()))
    assert math.isclose(score, run["geometric_score"], abs_tol=1e-10)
    assert rates["make_wood_pickaxe"] == run["wood_pickaxe_pct"]
    assert math.isclose(statistics.mean(e["return"] for e in rows), run["mean_return"], abs_tol=1e-10)
    assert run["retained_actions"] == 300000
for method, means in summary["method_means"].items():
    rows = [r for r in summary["runs"] if r["method"] == method]
    assert len(rows) == 3
    assert math.isclose(statistics.mean(r["geometric_score"] for r in rows), means["geometric_score"], abs_tol=1e-10)
print("Verified: 900 episodes, nine 22-achievement Scores, wood success, returns and method means.")
