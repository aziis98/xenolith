#!/usr/bin/env python3
import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


CASES = (
    ("pp512", 512, "prefill", "standard-xe-pp512-tg0", "standard-cpu-wide-pp512-tg0", "standard-vk-pp512-tg0"),
    ("pp2048", 2048, "prefill", "standard-xe-pp2048-tg0", "standard-cpu-wide-pp2048-tg0", "standard-vk-pp2048-tg0"),
    ("tg128", 128, "decode", "standard-xe-pp0-tg128", "standard-cpu-wide-pp0-tg128", "standard-vk-pp0-tg128"),
)


def samples_from_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def samples_from_runs(root):
    samples = []
    for _, _, _, *names in CASES:
        for name in names:
            run = root / name
            events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
            finishes = [event for event in events if event.get("event") == "guard_finish"]
            if len(finishes) != 1 or finishes[0].get("exit_code") != 0 or finishes[0].get("exit_signal") != 0 or finishes[0].get("aborted") is not False:
                raise ValueError(f"invalid guard result: {name}")
            for line in (run / "stdout.log").read_text().splitlines():
                event = json.loads(line)
                if event.get("event") == "measure" and event.get("phase") in ("prefill", "decode"):
                    samples.append({"run": name, "phase": event["phase"], "rep": event["rep"], "tokens": event["tokens"], "seconds": event["seconds"], "logged_tps": event["tps"]})
    return samples


def median_rate(samples, name, phase, tokens):
    selected = [sample for sample in samples if sample["run"] == name]
    if len(selected) != 3 or {int(sample["rep"]) for sample in selected} != {0, 1, 2}:
        raise ValueError(f"expected three repetitions: {name}")
    rates = []
    for sample in selected:
        if sample["phase"] != phase or int(sample["tokens"]) != tokens:
            raise ValueError(f"unexpected shape: {name}")
        seconds = Decimal(str(sample["seconds"]))
        if seconds <= 0:
            raise ValueError(f"invalid time: {name}")
        rate = Decimal(tokens) / seconds
        if abs(rate - Decimal(str(sample["logged_tps"]))) > Decimal("0.000001"):
            raise ValueError(f"logged rate does not match time: {name}")
        rates.append(rate)
    return sorted(rates)[1].quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def main():
    parser = argparse.ArgumentParser(description="Recompute pp/tg comparison medians")
    parser.add_argument("--runs", type=Path, help="directory containing new guarded run directories")
    args = parser.parse_args()
    bench = Path(__file__).resolve().parent
    samples = samples_from_runs(args.runs) if args.runs else samples_from_csv(bench / "compare_pp_tg-20260910.csv")
    if len(samples) != 27:
        raise ValueError(f"expected 27 measurements, got {len(samples)}")
    print("| Token/s | Xenolith | llama.cpp CPU | llama.cpp Vulkan |")
    print("|---|---:|---:|---:|")
    rendered = []
    for label, tokens, phase, *names in CASES:
        values = [median_rate(samples, name, phase, tokens) for name in names]
        row = f"| {label} | " + " | ".join(str(value).replace(".", ",") for value in values) + " |"
        rendered.append(row)
        print(row)
    if not args.runs:
        readme = (bench.parent / "README.md").read_text()
        for row in rendered:
            label, *values = [part.strip() for part in row.strip("|").split("|")]
            matching = next((line for line in readme.splitlines() if line.startswith(f"| {label} |")), None)
            if matching is None or [part.strip().strip("*") for part in matching.strip("|").split("|")[1:]] != values:
                raise ValueError(f"README differs from samples: {label}")


if __name__ == "__main__":
    main()
