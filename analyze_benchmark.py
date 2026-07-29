#!/usr/bin/env python3
"""Computes aggregate statistics from benchmark_results.json and writes
a final markdown report."""
import json
import statistics

RESULTS_PATH = "/home/alfaleus/projects/For Customers/benchmark_results.json"
REPORT_PATH = "/home/alfaleus/projects/For Customers/benchmark_report.md"

with open(RESULTS_PATH, encoding="utf-8") as f:
    results = [r for r in json.load(f) if r.get("recognition_time_sec") is not None]


def pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def group_by(key):
    groups = {}
    for r in results:
        groups.setdefault(r[key], []).append(r)
    return groups


def stats_for(rows):
    n = len(rows)
    passed = sum(1 for r in rows if r["pass"])
    wers = [r["wer"] for r in rows]
    cers = [r["cer"] for r in rows]
    sims = [r["similarity"] for r in rows]
    times = [r["recognition_time_sec"] for r in rows]
    confs = [r["confidence"] for r in rows if r["confidence"] is not None]
    return {
        "n": n,
        "pass": passed,
        "fail": n - passed,
        "accuracy_pct": pct(passed, n),
        "avg_wer": round(statistics.mean(wers), 4) if wers else None,
        "avg_cer": round(statistics.mean(cers), 4) if cers else None,
        "avg_similarity": round(statistics.mean(sims), 4) if sims else None,
        "avg_time": round(statistics.mean(times), 3) if times else None,
        "median_time": round(statistics.median(times), 3) if times else None,
        "avg_confidence": round(statistics.mean(confs), 4) if confs else None,
    }


overall = stats_for(results)
by_language = {lang: stats_for(rows) for lang, rows in sorted(group_by("language").items())}
by_category = {cat: stats_for(rows) for cat, rows in sorted(group_by("category").items())}
by_difficulty = {diff: stats_for(rows) for diff, rows in sorted(group_by("difficulty").items())}

# Language detection: every question was synthesized with the en-IN
# voice (see run_benchmark.py docstring), so "en-IN" detected is the
# phonetically correct answer for ALL 60 rows under this setup - this
# is NOT a measure of real Hindi/Telugu auto-detect accuracy.
detected_en_in = sum(1 for r in results if r["detected_language"] == "en-IN")
lang_detect_pct = pct(detected_en_in, len(results))

lines = []
lines.append("# STT Benchmark Report — Google Cloud STT v2 (Production Config)")
lines.append("")
lines.append("Generated from a TTS -> STT round-trip test: each of the 60 dataset "
              "questions was synthesized to speech with Google Cloud TTS (en-IN "
              "Chirp3-HD voice) and the resulting audio was fed through the exact "
              "same `transcribe()` path used in production "
              "(`companies/core/stt_google.py`, STT v2, `latest_long` model, "
              "`global` location, `language_codes=[en-IN, hi-IN, te-IN]`).")
lines.append("")
lines.append("## Important Caveats — Read Before Trusting These Numbers")
lines.append("")
lines.append("1. **No live caller was used.** There is no recording capability in "
              "this session, so real human speech was not available. This measures "
              "STT's raw recognition ability on clean, noise-free, single-speaker "
              "synthetic audio — real phone calls have background noise, telephone "
              "compression, natural disfluencies, and genuine speaker accents, all "
              "of which this test cannot capture. Expect real accuracy to be worse.")
lines.append("2. **All 60 questions were synthesized with the same en-IN English "
              "voice**, including the 25 Hindi and 25 Telugu (Romanized) questions — "
              "there is no native Hindi/Telugu script text in this dataset (by "
              "design) to feed a hi-IN/te-IN voice. The en-IN voice frequently "
              "mispronounces short/common words in its own stylized way (observed: "
              "\"to\" -> \"Tu\", \"the\" -> \"D\", \"are\" -> \"R\") — some WER/CER "
              "penalty below reflects **TTS mispronunciation**, not an STT failure, "
              "especially visible in the English results.")
lines.append("3. **Language-detection accuracy is not meaningful here** — since "
              "every clip was voiced in English phonetics, `en-IN` is the "
              "phonetically correct detection for all 60 rows regardless of the "
              "question's labeled language. This benchmark cannot validate real "
              "multilingual auto-detection; it only confirms the recognizer didn't "
              "spuriously flip to hi-IN/te-IN on English-voiced audio.")
lines.append("4. **Pass/Fail threshold:** WER <= 0.35 on normalized (lowercased, "
              "punctuation-stripped) text. This is a lenient bar chosen for "
              "phone-quality code-switched speech; tune once real-call data exists.")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Overall Results")
lines.append("")
lines.append(f"- **Overall Accuracy (Pass rate):** {overall['accuracy_pct']}% "
              f"({overall['pass']}/{overall['n']})")
lines.append(f"- **Failure Rate:** {pct(overall['fail'], overall['n'])}% "
              f"({overall['fail']}/{overall['n']})")
lines.append(f"- **Average WER:** {overall['avg_wer']}")
lines.append(f"- **Average CER:** {overall['avg_cer']}")
lines.append(f"- **Average Similarity Score:** {overall['avg_similarity']}")
lines.append(f"- **Average Recognition Time:** {overall['avg_time']}s")
lines.append(f"- **Median Recognition Time:** {overall['median_time']}s")
lines.append(f"- **Average Confidence:** {overall['avg_confidence']}")
lines.append(f"- **\"en-IN\" Detection Rate:** {lang_detect_pct}% "
              f"(expected under this all-English-voice setup — see caveat 3)")
lines.append("")
lines.append("## Accuracy Per Language")
lines.append("")
lines.append("| Language | N | Pass | Fail | Accuracy | Avg WER | Avg CER | Avg Time (s) | Avg Confidence |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for lang, s in by_language.items():
    lines.append(f"| {lang} | {s['n']} | {s['pass']} | {s['fail']} | {s['accuracy_pct']}% "
                 f"| {s['avg_wer']} | {s['avg_cer']} | {s['avg_time']} | {s['avg_confidence']} |")
lines.append("")
lines.append("## Accuracy Per Category")
lines.append("")
lines.append("| Category | N | Pass | Fail | Accuracy | Avg WER | Avg CER | Avg Time (s) |")
lines.append("|---|---|---|---|---|---|---|---|")
for cat, s in by_category.items():
    lines.append(f"| {cat} | {s['n']} | {s['pass']} | {s['fail']} | {s['accuracy_pct']}% "
                 f"| {s['avg_wer']} | {s['avg_cer']} | {s['avg_time']} |")
lines.append("")
lines.append("## Accuracy Per Difficulty")
lines.append("")
lines.append("| Difficulty | N | Pass | Fail | Accuracy | Avg WER | Avg CER | Avg Time (s) |")
lines.append("|---|---|---|---|---|---|---|---|")
for diff, s in by_difficulty.items():
    lines.append(f"| {diff} | {s['n']} | {s['pass']} | {s['fail']} | {s['accuracy_pct']}% "
                 f"| {s['avg_wer']} | {s['avg_cer']} | {s['avg_time']} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Full Results Table")
lines.append("")
lines.append("| ID | Language | Category | Difficulty | Expected | Recognized | WER | CER | Sim | Time(s) | Conf | Pass |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in results:
    exp = r["expected_text"].replace("|", "/")
    rec = (r["recognized_text"] or "(empty)").replace("|", "/")
    conf = r["confidence"] if r["confidence"] is not None else "n/a"
    lines.append(f"| {r['id']} | {r['language']} | {r['category']} | {r['difficulty']} "
                 f"| {exp} | {rec} | {r['wer']} | {r['cer']} | {r['similarity']} "
                 f"| {r['recognition_time_sec']} | {conf} | {'PASS' if r['pass'] else 'FAIL'} |")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Report written to {REPORT_PATH}")
print()
print(f"Overall accuracy: {overall['accuracy_pct']}% ({overall['pass']}/{overall['n']})")
print(f"By language:")
for lang, s in by_language.items():
    print(f"  {lang}: {s['accuracy_pct']}% ({s['pass']}/{s['n']}), avg WER {s['avg_wer']}, avg time {s['avg_time']}s")
