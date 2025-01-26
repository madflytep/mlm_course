import numpy as np


def calculate_accuracy(results):
    fields = list(results[0]["matches"].keys())
    counts = {f: 0 for f in fields}
    error_counts = {f: 0 for f in fields}
    full_match_count = 0
    for result in results:
        full_match = True
        for field in fields:
            counts[field] += 1
            if not result["matches"][field]:
                error_counts[field] += 1
                full_match = False
        if full_match:
            full_match_count += 1
    result = {"accuracy": 100 * round(full_match_count / len(results), 3)}
    result.update({f"accuracy_{f}": 100 * round(1 - error_counts[f] / counts[f], 3)
                   for f in fields})
    return result


def calculate_inference_time(results):
    times = [r["lm_execution_time"] for r in results]
    return {
        "mean": np.round(np.mean(times)).astype(int),
        "std": np.round(np.std(times)).astype(int),
        "min": np.min(times).astype(int),
        "max": np.max(times).astype(int)
    }
