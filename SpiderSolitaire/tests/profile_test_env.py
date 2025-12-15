import cProfile
import pstats
import json
from pathlib import Path
import sys

# Ensure repo root on path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from SpiderSolitaire.env.spider_env import SpiderEnv
from SpiderSolitaire.tests.test_env import evaluate_random_agent


def run_workload(n_episodes: int = 200, n_bins: int = 20):
	env = SpiderEnv(
		n_suits=1,
		n_actions_limit=100,
		rewards_policy={
			"discover_card": 4,
			"free pile": 16,
			"extend sequence": 1,
			"deal cards": -0.5,
		},
		mask_legal_actions=True,
	)
	return evaluate_random_agent(env, n_episodes=n_episodes, n_bins=n_bins)


def profile_to_files(sort_key=pstats.SortKey.CUMULATIVE, limit: int = 50):
	profile = cProfile.Profile()
	profile.enable()
	run_workload()
	profile.disable()

	stats_txt = Path(__file__).with_name("profile_test_env.stats")
	stats_json = Path(__file__).with_name("profile_test_env.json")

	# Human-readable stats
	with stats_txt.open("w", encoding="utf-8") as f:
		ps = pstats.Stats(profile, stream=f)
		ps.strip_dirs().sort_stats(sort_key).print_stats(limit)

	# JSON summary per function
	ps = pstats.Stats(profile)
	ps.strip_dirs().sort_stats(sort_key)
	data = []
	for (filename, line, funcname), stat in ps.stats.items():
		cc, nc, tt, ct, callers = stat
		data.append({
			"file": filename,
			"line": line,
			"func": funcname,
			"primitive_calls": cc,
			"total_calls": nc,
			"tottime": tt,
			"cumtime": ct,
		})
	data.sort(key=lambda x: x["cumtime"], reverse=True)
	with stats_json.open("w", encoding="utf-8") as jf:
		json.dump({"functions": data, "count": len(data)}, jf, indent=2)

	print(f"Wrote {stats_txt} and {stats_json}")


if __name__ == "__main__":
	profile_to_files()

 