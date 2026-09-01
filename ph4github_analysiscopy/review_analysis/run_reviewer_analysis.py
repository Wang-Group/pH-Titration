from __future__ import annotations

import ast
import csv
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader


REPO_ROOT = Path(r"E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy")
REVIEW_ANALYSIS = REPO_ROOT / "review_analysis"
REVIEW_OUTPUTS = REPO_ROOT / "review_outputs"
FIGURES_DIR = REVIEW_OUTPUTS / "figures"
TABLES_DIR = REVIEW_OUTPUTS / "tables"
BENCHMARK_DIR = REVIEW_OUTPUTS / "benchmarks"
LOGS_DIR = REVIEW_OUTPUTS / "logs"
PDF_DIR = REVIEW_OUTPUTS / "pdf_text"
MANUSCRIPT_DIR = REVIEW_OUTPUTS / "manuscript_text"

PDF_COPY = PDF_DIR / "SC-EDG-05-2026-003882_Round1_analysiscopy.pdf"
PDF_TEXT = PDF_DIR / "SC-EDG-05-2026-003882_Round1_analysiscopy.txt"
MANUSCRIPT_COPY = MANUSCRIPT_DIR / "manuscript_analysiscopy.docx"
MANUSCRIPT_TEXT = MANUSCRIPT_DIR / "manuscript_analysiscopy.txt"
SI_COPY = MANUSCRIPT_DIR / "si_analysiscopy.docx"
SI_TEXT = MANUSCRIPT_DIR / "si_analysiscopy.txt"

SOURCE_REPO_COPY = REPO_ROOT
SOURCE_MANUSCRIPT_COPY = Path(
    r"Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\manuscript\Hybrid Bayesian Inference and Reinforcement Learning for Autonomous pH Adjustment in Diverse Chemical Systems2.0_analysiscopy.docx"
)
SOURCE_SI_COPY = Path(
    r"Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\supplementary_information\SI-V19_analysiscopy.docx"
)

EXPERIMENT_SUMMARY = REPO_ROOT / "experiment_summary.csv"
DATA_DIR = REPO_ROOT / "data"
ALL_DATA_DIR = DATA_DIR / "all_data"

MIXED_ACID_DIR = ALL_DATA_DIR / "mixed_acid"
MILK_DIR = ALL_DATA_DIR / "milk"
SSA_DIR = ALL_DATA_DIR / "SSA"
WASTEWATER_DIR = ALL_DATA_DIR / "wastewater"

PLOT_NOTEBOOK = REPO_ROOT / "plot.ipynb"
MAIN_NOTEBOOK = REPO_ROOT / "main_code3.ipynb"
REPORT_PATH = REVIEW_ANALYSIS / "reviewer_analysis_report.md"
PRIORITY_PATH = REVIEW_ANALYSIS / "prioritized_task_list.md"


def ensure_dirs() -> None:
    for path in [REVIEW_ANALYSIS, FIGURES_DIR, TABLES_DIR, BENCHMARK_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def escape_md(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(escape_md(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep] + body)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_docx_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("\xa0", " ")
    )
    return text


def extract_text_artifacts() -> None:
    if PDF_COPY.exists():
        reader = PdfReader(str(PDF_COPY))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append("===== PAGE {} =====\n{}".format(idx, page_text))
        write_text(PDF_TEXT, "\n\n".join(pages))

    if MANUSCRIPT_COPY.exists():
        write_text(MANUSCRIPT_TEXT, parse_docx_text(MANUSCRIPT_COPY))
    if SI_COPY.exists():
        write_text(SI_TEXT, parse_docx_text(SI_COPY))


def parse_summary_block(path: Path) -> dict[str, float]:
    text = read_text(path)
    pattern = re.compile(
        r"Total experiments:\s*(?P<total>\d+)\s+"
        r"Successful experiments:\s*(?P<success>\d+)\s+"
        r"Success rate:\s*(?P<success_rate>[\d.]+)%\s+"
        r"Successful steps:\s*(?P<steps_mean>[\d.]+)\s+\+/-\s+(?P<steps_std>[\d.]+)\s+"
        r"Total steps:\s*(?P<total_steps>\d+)\s+"
        r"Total overshoots:\s*(?P<overshoots>\d+)\s+"
        r"Overshoot rate:\s*(?P<overshoot_rate>[\d.]+)%",
        re.S,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError("Could not parse summary block from {}".format(path))
    data = {key: float(value) for key, value in match.groupdict().items()}
    data["total"] = int(data["total"])
    data["success"] = int(data["success"])
    data["total_steps"] = int(data["total_steps"])
    data["overshoots"] = int(data["overshoots"])
    return data


def extract_notebook_cells(notebook_path: Path) -> list[dict[str, object]]:
    return json.loads(read_text(notebook_path)).get("cells", [])


def extract_cell_source(notebook_path: Path, index: int) -> str:
    cells = extract_notebook_cells(notebook_path)
    if index >= len(cells):
        raise IndexError("{} does not have cell {}".format(notebook_path, index))
    return "".join(cells[index].get("source", []))


def maybe_export_cell(notebook_path: Path, index: int, destination: Path) -> None:
    source = extract_cell_source(notebook_path, index)
    write_text(destination, source)


def parse_plot_metrics_from_notebook() -> list[dict[str, object]]:
    source = extract_cell_source(PLOT_NOTEBOOK, 22)
    maybe_export_cell(PLOT_NOTEBOOK, 22, LOGS_DIR / "plot_cell_22.py")
    models_match = re.search(r"models\s*=\s*\[(.*?)\]", source, re.S)
    success_match = re.search(r"success\s*=\s*\[(.*?)\]", source, re.S)
    overshoot_match = re.search(r"overshoot\s*=\s*\[(.*?)\]", source, re.S)
    steps_match = re.search(r"avg_steps\s*=\s*\[(.*?)\]", source, re.S)
    if not all([models_match, success_match, overshoot_match, steps_match]):
        raise ValueError("Could not parse hardcoded Figure/Table metrics from plot.ipynb cell 22")

    models = ast.literal_eval("[" + models_match.group(1) + "]")
    success = ast.literal_eval("[" + success_match.group(1) + "]")
    overshoot = ast.literal_eval("[" + overshoot_match.group(1) + "]")
    avg_steps = ast.literal_eval("[" + steps_match.group(1) + "]")
    rows = []
    for model, success_rate, overshoot_rate, steps in zip(models, success, overshoot, avg_steps):
        rows.append(
            {
                "source": "plot.ipynb cell 22",
                "algorithm": normalize_algorithm_name(model),
                "success_rate": float(success_rate),
                "avg_steps": float(steps),
                "overshoot_rate": float(overshoot_rate),
                "evidence": str(PLOT_NOTEBOOK),
            }
        )
    return rows


def normalize_algorithm_name(name: str) -> str:
    lowered = name.lower()
    if "bayes" in lowered:
        return "Bayesian"
    if "imit" in lowered:
        return "Imitation"
    if "rein" in lowered:
        return "Reinforcement"
    if "pid" in lowered:
        return "PID"
    if "human" in lowered:
        return "Human"
    return name


def build_reviewer_mapping() -> list[dict[str, object]]:
    return [
        {
            "Reviewer comment ID": "1a",
            "Short summary": "Clarify what is novel beyond prior RL/Bayesian pH-control work.",
            "Why it needs analysis/code": "Requires direct method comparison and exact local evidence for Bayesian vs IL vs RL behavior.",
            "Relevant local files": "; ".join(
                [
                    str(MAIN_NOTEBOOK),
                    str(DATA_DIR / "bayesian.txt"),
                    str(DATA_DIR / "m_network.txt"),
                    str(DATA_DIR / "reinforced_network.txt"),
                    str(MANUSCRIPT_TEXT),
                ]
            ),
            "Proposed deliverable": "Trusted simulation-metrics table plus method-consistency audit.",
        },
        {
            "Reviewer comment ID": "1b",
            "Short summary": "Substantiate interpretability claims for Bayesian control.",
            "Why it needs analysis/code": "Needs posterior/uncertainty traces from the local Bayesian update code.",
            "Relevant local files": "; ".join([str(MAIN_NOTEBOOK), str(EXPERIMENT_SUMMARY)]),
            "Proposed deliverable": "Posterior-uncertainty shrinkage figure and limitations note.",
        },
        {
            "Reviewer comment ID": "1c",
            "Short summary": "Explain Bayesian policy advantage over a neural network trained directly on simulation.",
            "Why it needs analysis/code": "Requires inspecting how the imitation dataset is generated and whether a direct simulation-only baseline exists.",
            "Relevant local files": "; ".join([str(MAIN_NOTEBOOK), str(MANUSCRIPT_TEXT), str(SI_TEXT)]),
            "Proposed deliverable": "Baseline-status note plus a minimal direct-supervised experiment proposal.",
        },
        {
            "Reviewer comment ID": "3",
            "Short summary": "Benchmark computational cost against the 20 s experimental cycle.",
            "Why it needs analysis/code": "Requires timing measurements on the local Bayesian code path and the learned-policy architecture.",
            "Relevant local files": "; ".join([str(MAIN_NOTEBOOK), str(EXPERIMENT_SUMMARY), str(SI_TEXT)]),
            "Proposed deliverable": "Timing benchmark CSV and manuscript-ready comparison paragraph.",
        },
        {
            "Reviewer comment ID": "4",
            "Short summary": "Assess whether the human advantage in Figure 2 is just small-sample noise.",
            "Why it needs analysis/code": "Requires reconstructing the physical benchmark mapping and counting the true sample size.",
            "Relevant local files": "; ".join([str(MIXED_ACID_DIR), str(MANUSCRIPT_TEXT), str(SI_TEXT)]),
            "Proposed deliverable": "Physical-step audit with an explicit n=4 limitation statement.",
        },
        {
            "Reviewer comment ID": "4a",
            "Short summary": "Evaluate an expert-rule or PID-style baseline.",
            "Why it needs analysis/code": "Requires locating, running, and summarizing the baseline code.",
            "Relevant local files": "; ".join([str(MAIN_NOTEBOOK), str(EXPERIMENT_SUMMARY), str(DATA_DIR / "PIDexperiment.txt")]),
            "Proposed deliverable": "Recomputed PID summary and comparison table.",
        },
        {
            "Reviewer comment ID": "6a",
            "Short summary": "Fix the Figure 2 x-axis labeling pattern.",
            "Why it needs analysis/code": "Requires auditing notebook plotting logic and producing a revised draft figure script.",
            "Relevant local files": "; ".join([str(PLOT_NOTEBOOK), str(MIXED_ACID_DIR)]),
            "Proposed deliverable": "Revised Figure 2 candidate with step ticks every 2 steps plus final step.",
        },
        {
            "Reviewer comment ID": "6b",
            "Short summary": "Remove the errant line in Figure 2.",
            "Why it needs analysis/code": "Needs a clean plotting implementation with controlled line/marker rendering.",
            "Relevant local files": "; ".join([str(PLOT_NOTEBOOK), str(MIXED_ACID_DIR)]),
            "Proposed deliverable": "Revised Figure 2 candidate without stray connecting artifacts.",
        },
        {
            "Reviewer comment ID": "6c",
            "Short summary": "Use consistent pH axes within each row of Figure 2.",
            "Why it needs analysis/code": "Needs a row-shared y-limit plotting pass.",
            "Relevant local files": "; ".join([str(PLOT_NOTEBOOK), str(MIXED_ACID_DIR)]),
            "Proposed deliverable": "Revised Figure 2 candidate with row-consistent pH limits.",
        },
        {
            "Reviewer comment ID": "7",
            "Short summary": "Provide titration-curve plots for the example buffered systems.",
            "Why it needs analysis/code": "Needs SI-ready figure generation from local physical datasets or local models.",
            "Relevant local files": "; ".join([str(MIXED_ACID_DIR), str(SI_TEXT)]),
            "Proposed deliverable": "Effective empirical titration-curve figure and an explicit note on the lack of exact theoretical inputs.",
        },
        {
            "Reviewer comment ID": "8a",
            "Short summary": "Replace unquantified 'significantly' language with numbers.",
            "Why it needs analysis/code": "Needs exact effect sizes from the trusted metrics audit.",
            "Relevant local files": "; ".join([str(DATA_DIR / "bayesian.txt"), str(DATA_DIR / "m_network.txt"), str(DATA_DIR / "reinforced_network.txt")]),
            "Proposed deliverable": "Percentage-point and step-count deltas ready for manuscript wording.",
        },
        {
            "Reviewer comment ID": "9",
            "Short summary": "Clarify what the physical dosing increments/resolution really were.",
            "Why it needs analysis/code": "Needs exact action-volume ranges from code and physical logs.",
            "Relevant local files": "; ".join([str(PLOT_NOTEBOOK), str(MANUSCRIPT_TEXT), str(ALL_DATA_DIR)]),
            "Proposed deliverable": "Simulation-vs-physical volume-range summary table.",
        },
        {
            "Reviewer comment ID": "10",
            "Short summary": "Clarify which policy was used in Experiments 3 and 4.",
            "Why it needs analysis/code": "Needs a manuscript/SI policy-usage audit against the local text and datasets.",
            "Relevant local files": "; ".join([str(MANUSCRIPT_TEXT), str(SI_TEXT), str(ALL_DATA_DIR)]),
            "Proposed deliverable": "Policy-usage note with exact local text evidence.",
        },
        {
            "Reviewer comment ID": "11",
            "Short summary": "Report input volume ranges for training and experiments.",
            "Why it needs analysis/code": "Needs extraction from notebook code and physical experiment files.",
            "Relevant local files": "; ".join([str(MAIN_NOTEBOOK), str(PLOT_NOTEBOOK), str(ALL_DATA_DIR)]),
            "Proposed deliverable": "Action-space summary table with dataset-specific min/max values.",
        },
        {
            "Reviewer comment ID": "12",
            "Short summary": "Single giant notebook hinders reproducibility.",
            "Why it needs analysis/code": "Needs a practical reproducibility checklist tied to the current repository state.",
            "Relevant local files": "; ".join([str(MAIN_NOTEBOOK), str(REPO_ROOT / "README.md")]),
            "Proposed deliverable": "Revision-ready reproducibility checklist.",
        },
        {
            "Reviewer comment ID": "13",
            "Short summary": "The repository README and environment documentation are incomplete.",
            "Why it needs analysis/code": "Needs an actionable list of missing documentation and environment files.",
            "Relevant local files": "; ".join([str(REPO_ROOT / "README.md"), str(MAIN_NOTEBOOK)]),
            "Proposed deliverable": "README/requirements/notebook-extraction checklist.",
        },
    ]


def parse_simulation_metrics() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    raw_sources = {
        "Bayesian": DATA_DIR / "bayesian.txt",
        "Imitation": DATA_DIR / "m_network.txt",
        "Reinforcement": DATA_DIR / "reinforced_network.txt",
        "PID": DATA_DIR / "PIDexperiment.txt",
    }
    raw_metrics: dict[str, dict[str, float]] = {}
    rows = []
    for algorithm, path in raw_sources.items():
        summary = parse_summary_block(path)
        raw_metrics[algorithm] = summary
        rows.append(
            {
                "algorithm": algorithm,
                "source_type": "raw_result_log",
                "source": str(path),
                "success_rate": summary["success_rate"],
                "avg_successful_steps": summary["steps_mean"],
                "successful_steps_std": summary["steps_std"],
                "total_steps": summary["total_steps"],
                "total_overshoots": summary["overshoots"],
                "overshoot_rate": summary["overshoot_rate"],
                "successful_experiments": summary["success"],
                "total_experiments": summary["total"],
                "trusted": "yes",
                "notes": "Parsed from terminal-style summary block at the end of the raw result file.",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "policy_metrics_raw_summary.csv", index=False, encoding="utf-8-sig")
    return frame, raw_metrics


def build_metrics_claim_audit(raw_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    claim_rows = []

    plot_rows = parse_plot_metrics_from_notebook()
    for row in plot_rows:
        algorithm = row["algorithm"]
        trusted = raw_metrics[algorithm]
        claim_rows.append(
            {
                "source_group": "plot_notebook_hardcoded",
                "source": row["source"],
                "algorithm": algorithm,
                "claimed_success_rate": row["success_rate"],
                "claimed_avg_successful_steps": row["avg_steps"],
                "claimed_overshoot_rate": row["overshoot_rate"],
                "trusted_success_rate": trusted["success_rate"],
                "trusted_avg_successful_steps": trusted["steps_mean"],
                "trusted_overshoot_rate": trusted["overshoot_rate"],
                "delta_success_rate_pp": round(row["success_rate"] - trusted["success_rate"], 2),
                "delta_avg_successful_steps": round(row["avg_steps"] - trusted["steps_mean"], 2),
                "delta_overshoot_rate_pp": round(row["overshoot_rate"] - trusted["overshoot_rate"], 2),
                "evidence": row["evidence"],
                "status": "mismatch",
            }
        )

    manuscript_claims = [
        {"algorithm": "Bayesian", "success_rate": 94.2, "avg_steps": 12.73, "overshoot_rate": 41.84},
        {"algorithm": "Imitation", "success_rate": 93.77, "avg_steps": 10.22, "overshoot_rate": 34.41},
        {"algorithm": "Reinforcement", "success_rate": 94.27, "avg_steps": 10.21, "overshoot_rate": 30.55},
    ]
    for claim in manuscript_claims:
        trusted = raw_metrics[claim["algorithm"]]
        claim_rows.append(
            {
                "source_group": "manuscript_table1_and_narrative",
                "source": str(MANUSCRIPT_TEXT),
                "algorithm": claim["algorithm"],
                "claimed_success_rate": claim["success_rate"],
                "claimed_avg_successful_steps": claim["avg_steps"],
                "claimed_overshoot_rate": claim["overshoot_rate"],
                "trusted_success_rate": trusted["success_rate"],
                "trusted_avg_successful_steps": trusted["steps_mean"],
                "trusted_overshoot_rate": trusted["overshoot_rate"],
                "delta_success_rate_pp": round(claim["success_rate"] - trusted["success_rate"], 2),
                "delta_avg_successful_steps": round(claim["avg_steps"] - trusted["steps_mean"], 2),
                "delta_overshoot_rate_pp": round(claim["overshoot_rate"] - trusted["overshoot_rate"], 2),
                "evidence": str(MANUSCRIPT_TEXT),
                "status": "mismatch",
            }
        )

    si_updated_claims = [
        {"algorithm": "Bayesian", "success_rate": 90.3, "avg_steps": 9.0, "overshoot_rate": 44.3},
        {"algorithm": "Imitation", "success_rate": 93.9, "avg_steps": 10.0, "overshoot_rate": 31.3},
        {"algorithm": "Reinforcement", "success_rate": 94.3, "avg_steps": 10.0, "overshoot_rate": 31.4},
        {"algorithm": "PID", "success_rate": 76.2, "avg_steps": 24.0, "overshoot_rate": 29.9},
    ]
    for claim in si_updated_claims:
        trusted = raw_metrics[claim["algorithm"]]
        claim_rows.append(
            {
                "source_group": "si_updated_rounding",
                "source": str(SI_TEXT),
                "algorithm": claim["algorithm"],
                "claimed_success_rate": claim["success_rate"],
                "claimed_avg_successful_steps": claim["avg_steps"],
                "claimed_overshoot_rate": claim["overshoot_rate"],
                "trusted_success_rate": trusted["success_rate"],
                "trusted_avg_successful_steps": trusted["steps_mean"],
                "trusted_overshoot_rate": trusted["overshoot_rate"],
                "delta_success_rate_pp": round(claim["success_rate"] - trusted["success_rate"], 2),
                "delta_avg_successful_steps": round(claim["avg_steps"] - trusted["steps_mean"], 2),
                "delta_overshoot_rate_pp": round(claim["overshoot_rate"] - trusted["overshoot_rate"], 2),
                "evidence": str(SI_TEXT),
                "status": "rounded_match" if claim["algorithm"] != "PID" else "rounded_match",
            }
        )

    si_stale_claims = [
        {"algorithm": "Bayesian", "success_rate": 94.2, "avg_steps": 13.73, "overshoot_rate": 39.99},
        {"algorithm": "Imitation", "success_rate": 93.97, "avg_steps": 10.2, "overshoot_rate": 28.44},
        {"algorithm": "Reinforcement", "success_rate": 94.327, "avg_steps": 10.1, "overshoot_rate": 28.67},
        {"algorithm": "PID", "success_rate": 29.703, "avg_steps": 44.19, "overshoot_rate": 24.4},
    ]
    for claim in si_stale_claims:
        trusted = raw_metrics[claim["algorithm"]]
        claim_rows.append(
            {
                "source_group": "si_stale_paragraph",
                "source": str(SI_TEXT),
                "algorithm": claim["algorithm"],
                "claimed_success_rate": claim["success_rate"],
                "claimed_avg_successful_steps": claim["avg_steps"],
                "claimed_overshoot_rate": claim["overshoot_rate"],
                "trusted_success_rate": trusted["success_rate"],
                "trusted_avg_successful_steps": trusted["steps_mean"],
                "trusted_overshoot_rate": trusted["overshoot_rate"],
                "delta_success_rate_pp": round(claim["success_rate"] - trusted["success_rate"], 2),
                "delta_avg_successful_steps": round(claim["avg_steps"] - trusted["steps_mean"], 2),
                "delta_overshoot_rate_pp": round(claim["overshoot_rate"] - trusted["overshoot_rate"], 2),
                "evidence": str(SI_TEXT),
                "status": "mismatch",
            }
        )

    audit = pd.DataFrame(claim_rows)
    audit.to_csv(TABLES_DIR / "metrics_claim_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def build_method_consistency_audit() -> pd.DataFrame:
    maybe_export_cell(MAIN_NOTEBOOK, 3, LOGS_DIR / "main_code3_cell_3.py")
    maybe_export_cell(MAIN_NOTEBOOK, 5, LOGS_DIR / "main_code3_cell_5.py")
    maybe_export_cell(MAIN_NOTEBOOK, 7, LOGS_DIR / "main_code3_cell_7.py")
    maybe_export_cell(MAIN_NOTEBOOK, 9, LOGS_DIR / "main_code3_cell_9.py")
    maybe_export_cell(MAIN_NOTEBOOK, 19, LOGS_DIR / "main_code3_cell_19.py")
    maybe_export_cell(MAIN_NOTEBOOK, 33, LOGS_DIR / "main_code3_cell_33.py")

    rows = [
        {
            "topic": "imitation_architecture",
            "text_claim": "Manuscript says the imitation MLP has two hidden layers of 64 units and is trained with MSE.",
            "code_evidence": "main_code3.ipynb cells 5 and 7 define a 5->256->256->1000 classifier trained with CrossEntropyLoss.",
            "local_files": "{}; {}".format(MANUSCRIPT_TEXT, LOGS_DIR / "main_code3_cell_5.py"),
            "impact": "Method description is inconsistent with the local implementation.",
        },
        {
            "topic": "reinforcement_algorithm",
            "text_claim": "Manuscript says PPO with 50,000 episodes; SI says REINFORCE but also contains 500/1000-episode text.",
            "code_evidence": "main_code3.ipynb cell 9 implements REINFORCE and trains for 500 episodes by default.",
            "local_files": "{}; {}; {}".format(MANUSCRIPT_TEXT, SI_TEXT, LOGS_DIR / "main_code3_cell_9.py"),
            "impact": "The RL method section needs a full consistency pass before revision.",
        },
        {
            "topic": "imitation_dataset_source",
            "text_claim": "Text sometimes frames imitation as simulation-only data without distinguishing the Bayesian teacher.",
            "code_evidence": "main_code3.ipynb cell 3 generates state-action pairs by calling the Bayesian controller (select_best_action and update_posteriors).",
            "local_files": "{}; {}".format(MANUSCRIPT_TEXT, LOGS_DIR / "main_code3_cell_3.py"),
            "impact": "The direct-simulated-data baseline requested by the reviewer does not already exist as a distinct local model.",
        },
        {
            "topic": "volume_range",
            "text_claim": "Text gives imprecise or mixed volume-range language.",
            "code_evidence": "Bayesian controller uses 0.01-9.99 mL candidate volumes; learned policies use 0.01-10.00 mL discrete bins; PID caps at 3.00 mL.",
            "local_files": "{}; {}; {}".format(LOGS_DIR / "main_code3_cell_1.py", LOGS_DIR / "main_code3_cell_5.py", LOGS_DIR / "main_code3_cell_33.py"),
            "impact": "The methods section should separate Bayesian simulation, learned-policy action bins, and physical dosing limits.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "method_consistency_audit.csv", index=False, encoding="utf-8-sig")
    return frame


def load_stepwise_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gbk")
    else:
        df = pd.read_excel(path)

    if "actual_volume" in df.columns:
        volume_col = "actual_volume"
    elif "volume" in df.columns:
        volume_col = "volume"
    elif "recommended_volume" in df.columns:
        volume_col = "recommended_volume"
    else:
        raise ValueError("No recognizable volume column in {}".format(path))

    df = df.copy()
    df["volume_for_analysis"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    df["reagent_norm"] = df["reagent"].astype(str).str.strip().str.lower()

    if "step" not in df.columns or df["step"].isnull().any():
        df.insert(0, "step", np.arange(1, len(df) + 1, dtype=int))
    df["step"] = df["step"].astype(int)

    step0 = {
        "step": 0,
        "reagent": "none",
        "reagent_norm": "none",
        "volume_for_analysis": 0.0,
        "pH_before": float(df.loc[0, "pH_before"]),
        "pH_after": float(df.loc[0, "pH_before"]),
        "pH_change": 0.0,
    }
    merged = pd.concat([pd.DataFrame([step0]), df], ignore_index=True, sort=False)
    merged["signed_volume_mL"] = np.where(
        merged["reagent_norm"].str.contains("base"),
        merged["volume_for_analysis"],
        np.where(merged["reagent_norm"].str.contains("acid"), -merged["volume_for_analysis"], 0.0),
    )
    merged["cumulative_total_volume_mL"] = merged["volume_for_analysis"].cumsum()
    merged["cumulative_signed_volume_mL"] = merged["signed_volume_mL"].cumsum()
    return merged


def summarize_volume_ranges() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "category": "simulation",
            "dataset": "Bayesian controller action space",
            "files": str(LOGS_DIR / "main_code3_cell_1.py"),
            "volume_column": "addition_volumes",
            "min_mL": 0.01,
            "max_mL": 9.99,
            "resolution_mL": 0.01,
            "note": "0.01-9.99 mL over four reagents; secondary 0.01 M reagents are available after oscillation.",
        },
        {
            "category": "simulation",
            "dataset": "Imitation policy action bins",
            "files": str(LOGS_DIR / "main_code3_cell_5.py"),
            "volume_column": "discrete_volumes",
            "min_mL": 0.01,
            "max_mL": 10.00,
            "resolution_mL": 0.01,
            "note": "1000 discrete classes from 0.01 to 10.00 mL.",
        },
        {
            "category": "simulation",
            "dataset": "Reinforcement policy action bins",
            "files": str(LOGS_DIR / "main_code3_cell_9.py"),
            "volume_column": "discrete_volumes",
            "min_mL": 0.01,
            "max_mL": 10.00,
            "resolution_mL": 0.01,
            "note": "Same 5->256->256->1000 action head as imitation learning.",
        },
        {
            "category": "simulation",
            "dataset": "Adaptive PID baseline output",
            "files": str(LOGS_DIR / "main_code3_cell_33.py"),
            "volume_column": "controller output",
            "min_mL": 0.01,
            "max_mL": 3.00,
            "resolution_mL": 0.001,
            "note": "PID volume is clamped to 0.01-3.00 mL and rounded to 0.001 mL in code.",
        },
    ]

    file_rows = []
    datasets = {
        "mixed_acid": sorted(MIXED_ACID_DIR.glob("*.xlsx")),
        "milk": sorted(MILK_DIR.glob("*.xlsx")),
        "ssa": sorted(path for path in SSA_DIR.glob("SSA*.csv") if "UV-Vis" not in path.name),
        "wastewater": sorted(WASTEWATER_DIR.glob("WasteWater*.csv")),
    }

    for dataset_name, files in datasets.items():
        for path in files:
            df = load_stepwise_file(path)
            positive = df.loc[df["volume_for_analysis"] > 0, "volume_for_analysis"]
            file_rows.append(
                {
                    "dataset": dataset_name,
                    "file": path.name,
                    "rows": len(df) - 1,
                    "min_observed_mL": float(positive.min()) if not positive.empty else 0.0,
                    "max_observed_mL": float(positive.max()) if not positive.empty else 0.0,
                    "volume_column_used": (
                        "actual_volume"
                        if "actual_volume" in df.columns
                        else "volume"
                        if "volume" in df.columns
                        else "recommended_volume"
                    ),
                }
            )

        all_values = pd.concat([load_stepwise_file(path)["volume_for_analysis"] for path in files], ignore_index=True)
        positive = all_values[all_values > 0]
        rows.append(
            {
                "category": "physical",
                "dataset": dataset_name,
                "files": "; ".join(str(path) for path in files),
                "volume_column": "dataset-specific",
                "min_mL": float(positive.min()) if not positive.empty else 0.0,
                "max_mL": float(positive.max()) if not positive.empty else 0.0,
                "resolution_mL": float(positive.sort_values().diff().abs().replace(0, np.nan).min())
                if len(positive) > 1
                else 0.0,
                "note": build_physical_volume_note(dataset_name),
            }
        )

    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(file_rows)
    summary.to_csv(TABLES_DIR / "volume_range_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(TABLES_DIR / "volume_range_file_detail.csv", index=False, encoding="utf-8-sig")
    return summary, detail


def build_physical_volume_note(dataset_name: str) -> str:
    notes = {
        "mixed_acid": "Volume is taken from the raw 'volume' column in the 12 mixed-acid benchmark files; these are the Figure 2-style physical runs.",
        "milk": "Single milk file with raw 'volume' entries.",
        "ssa": "Actual volume differs from recommended_volume because acid additions are corrected by the neutralization factor; early base additions are capped at 5.0 mL.",
        "wastewater": "Same recommended-vs-actual behavior as SSA, with explicit 5.0 mL truncation in the experimental files.",
    }
    return notes[dataset_name]


def mixed_acid_sort_key(path: Path) -> tuple[int, int]:
    first, second = path.stem.split("-")
    return int(first), int(second)


def summarize_mixed_acid_controller_mapping() -> pd.DataFrame:
    controller_map = {"1": "Bayesian", "2": "Reinforcement", "3": "Human"}
    rows = []
    for path in sorted(MIXED_ACID_DIR.glob("*.xlsx"), key=mixed_acid_sort_key):
        mixture, suffix = path.stem.split("-")
        df = load_stepwise_file(path)
        rows.append(
            {
                "mixture": int(mixture),
                "controller": controller_map[suffix],
                "file": path.name,
                "steps": int(df["step"].max()),
                "final_pH": float(df.loc[df["step"].idxmax(), "pH_after"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "mixed_acid_controller_steps.csv", index=False, encoding="utf-8-sig")
    return frame


def make_step_ticks(max_step: int) -> list[int]:
    ticks = [0]
    ticks.extend(range(2, max_step + 1, 2))
    if max_step not in ticks:
        ticks.append(max_step)
    return sorted(set(ticks))


def plot_figure2_candidate() -> list[Path]:
    controller_order = ["Bayesian", "Reinforcement", "Human"]
    controller_suffix = {"Bayesian": "1", "Reinforcement": "2", "Human": "3"}
    color_map = {"base": "#d1495b", "acid": "#2d6a4f", "none": "#9a8c98"}
    marker_map = {"base": "^", "acid": "v", "none": "o"}

    fig, axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
    for mixture_id in range(1, 5):
        series = {}
        y_values = []
        for controller in controller_order:
            path = MIXED_ACID_DIR / "{}-{}.xlsx".format(mixture_id, controller_suffix[controller])
            df = load_stepwise_file(path)
            series[controller] = df
            y_values.extend(df["pH_after"].tolist())

        y_min = max(0.0, min(y_values) - 0.4)
        y_max = min(14.0, max(y_values) + 0.4)

        for col, controller in enumerate(controller_order):
            ax = axes[mixture_id - 1, col]
            df = series[controller]
            ax.plot(df["step"], df["pH_after"], color="#4a4e69", linewidth=1.2, zorder=1)
            for reagent in ["none", "acid", "base"]:
                mask = df["reagent_norm"] == reagent
                if mask.any():
                    ax.scatter(
                        df.loc[mask, "step"],
                        df.loc[mask, "pH_after"],
                        color=color_map[reagent],
                        marker=marker_map[reagent],
                        s=45,
                        zorder=3,
                    )
            ax.axhline(6.0, color="#1d3557", linestyle="--", linewidth=1.0, zorder=2)
            max_step = int(df["step"].max())
            ax.set_xticks(make_step_ticks(max_step))
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(-0.5, max_step + 0.5)
            ax.grid(alpha=0.2, linewidth=0.5)
            if mixture_id == 1:
                ax.set_title(controller, fontsize=11)
            if col == 0:
                ax.set_ylabel("Mixture {} pH".format(mixture_id))
            if mixture_id == 4:
                ax.set_xlabel("Step")

    handles = [
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=color_map["base"], markersize=8, label="Base addition"),
        plt.Line2D([0], [0], marker="v", color="none", markerfacecolor=color_map["acid"], markersize=8, label="Acid addition"),
        plt.Line2D([0], [0], color="#1d3557", linestyle="--", label="Target pH 6.0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)

    png_path = FIGURES_DIR / "figure2_candidate_revised.png"
    svg_path = FIGURES_DIR / "figure2_candidate_revised.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def plot_effective_titration_curves() -> tuple[list[Path], pd.DataFrame]:
    controller_map = {"1": "Bayesian", "2": "Reinforcement", "3": "Human"}
    color_map = {"Bayesian": "#355070", "Reinforcement": "#6d597a", "Human": "#b56576"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    records = []
    for mixture_id in range(1, 5):
        ax = axes[(mixture_id - 1) // 2, (mixture_id - 1) % 2]
        for suffix, controller in controller_map.items():
            path = MIXED_ACID_DIR / "{}-{}.xlsx".format(mixture_id, suffix)
            df = load_stepwise_file(path)
            ax.plot(
                df["cumulative_signed_volume_mL"],
                df["pH_after"],
                marker="o",
                markersize=4,
                linewidth=1.2,
                label=controller,
                color=color_map[controller],
            )
            for _, row in df.iterrows():
                records.append(
                    {
                        "mixture": mixture_id,
                        "controller": controller,
                        "step": int(row["step"]),
                        "cumulative_signed_volume_mL": float(row["cumulative_signed_volume_mL"]),
                        "cumulative_total_volume_mL": float(row["cumulative_total_volume_mL"]),
                        "pH_after": float(row["pH_after"]),
                        "file": path.name,
                    }
                )

        ax.axhline(6.0, color="#1d3557", linestyle="--", linewidth=0.9)
        ax.set_title("Mixture {}".format(mixture_id))
        ax.set_xlabel("Cumulative signed titrant volume (mL)")
        ax.set_ylabel("pH")
        ax.grid(alpha=0.2, linewidth=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    png_path = FIGURES_DIR / "effective_titration_curves_mixed_acid.png"
    svg_path = FIGURES_DIR / "effective_titration_curves_mixed_acid.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    frame = pd.DataFrame(records)
    frame.to_csv(TABLES_DIR / "effective_titration_curves_mixed_acid.csv", index=False, encoding="utf-8-sig")
    return [png_path, svg_path], frame


TITRANT_CONC = 0.1
INITIAL_ACID_VOL = 11.0
MAX_STEPS = 50
SUCCESS_THRESHOLD = 0.1
MIN_PID_VOLUME = 0.01
MAX_PID_VOLUME = 3.0


def parse_pkas(raw_value: str) -> list[float]:
    parsed = ast.literal_eval(str(raw_value))
    if isinstance(parsed, list):
        return [float(value) for value in parsed]
    return [float(parsed)]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def get_acid_charge_factor(pH: float, pKas: list[float]) -> float:
    hydrogen = 10 ** (-pH)
    Kas = [10 ** (-pk) for pk in sorted(pKas)]
    coeffs = [1.0]
    current = 1.0
    for K in Kas:
        current *= K
        coeffs.append(current)
    terms = [coeffs[index] * (hydrogen ** (len(Kas) - index)) for index in range(len(Kas) + 1)]
    denominator = sum(terms)
    return sum(index * terms[index] for index in range(len(Kas) + 1)) / denominator


def pid_charge_balance_equation(pH: float, c_A: float, c_Na: float, c_HCl: float, pKas: list[float]) -> float:
    hydrogen = 10 ** (-pH)
    hydroxide = 1e-14 / hydrogen
    acid_negative_charge = c_A * get_acid_charge_factor(pH, pKas)
    return hydrogen + c_Na - hydroxide - c_HCl - acid_negative_charge


def pid_solve_ph(base_vol: float, acid_vol: float, pKas: list[float]) -> float:
    total_vol_l = (INITIAL_ACID_VOL + base_vol + acid_vol) / 1000
    c_A = (INITIAL_ACID_VOL * 0.1 / 1000) / total_vol_l
    c_Na = (base_vol * TITRANT_CONC / 1000) / total_vol_l
    c_HCl = (acid_vol * TITRANT_CONC / 1000) / total_vol_l

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if pid_charge_balance_equation(mid, c_A, c_Na, c_HCl, pKas) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


class PIDTitrationEnv:
    def __init__(self) -> None:
        self.pKas: list[float] = []
        self.target_ph = 0.0
        self.base_added = 0.0
        self.acid_added = 0.0
        self.current_ph = 0.0
        self.steps = 0

    def reset_state(self, pKas: list[float], target_ph: float) -> float:
        self.pKas = pKas
        self.target_ph = target_ph
        self.base_added = 0.0
        self.acid_added = 0.0
        self.current_ph = pid_solve_ph(0, 0, self.pKas)
        self.steps = 0
        return self.current_ph

    def step(self, volume: float) -> tuple[float, str, float, bool]:
        previous_ph = self.current_ph
        reagent = "Base" if self.current_ph < self.target_ph else "Acid"
        if reagent == "Base":
            self.base_added += volume
        else:
            self.acid_added += volume
        self.current_ph = pid_solve_ph(self.base_added, self.acid_added, self.pKas)
        self.steps += 1
        overshoot = (
            (previous_ph < self.target_ph and self.current_ph > self.target_ph)
            or (previous_ph > self.target_ph and self.current_ph < self.target_ph)
        )
        return self.current_ph, reagent, volume, overshoot


class AdaptivePIDController:
    def __init__(
        self,
        kp: float = 0.32,
        ki: float = 0.012,
        kd: float = 0.08,
        integral_limit: float = 12.0,
        output_limit: float = MAX_PID_VOLUME,
        overshoot_decay: float = 0.10,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.overshoot_decay = overshoot_decay
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error: float | None = None

    def get_volume(self, current_ph: float, target_ph: float) -> float:
        error = target_ph - current_ph
        if self.previous_error is not None and error * self.previous_error < 0:
            self.integral *= self.overshoot_decay
        self.integral = clamp(self.integral + error, -self.integral_limit, self.integral_limit)
        derivative = 0.0 if self.previous_error is None else error - self.previous_error
        self.previous_error = error
        signal = self.kp * error + self.ki * self.integral + self.kd * derivative
        volume = clamp(abs(signal), MIN_PID_VOLUME, self.output_limit)
        return round(volume, 3)


def rerun_pid_summary() -> pd.DataFrame:
    env = PIDTitrationEnv()
    controller = AdaptivePIDController()
    rows = pd.read_csv(EXPERIMENT_SUMMARY)

    success_steps = []
    total_steps = 0
    total_overshoots = 0
    success_count = 0

    per_experiment = []
    for _, row in rows.iterrows():
        pKas = parse_pkas(row["Acid_Params"])
        target_ph = float(row["Target_pH"])
        current_ph = env.reset_state(pKas, target_ph)
        controller.reset()
        overshoots = 0

        while True:
            volume = controller.get_volume(current_ph, target_ph)
            current_ph, reagent, used_volume, overshot = env.step(volume)
            if overshot:
                overshoots += 1
            if abs(current_ph - target_ph) <= SUCCESS_THRESHOLD:
                success = True
                break
            if env.steps >= MAX_STEPS:
                success = False
                break

        total_steps += env.steps
        total_overshoots += overshoots
        if success:
            success_count += 1
            success_steps.append(env.steps)

        per_experiment.append(
            {
                "experiment": int(row["Experiment"]),
                "acid_type": row["Acid_Type"],
                "target_pH": target_ph,
                "final_pH": current_ph,
                "steps": env.steps,
                "overshoots": overshoots,
                "success": success,
            }
        )

    summary = pd.DataFrame(
        [
            {
                "algorithm": "PID",
                "rerun_success_rate": round(success_count / len(rows) * 100, 2),
                "rerun_avg_successful_steps": round(statistics.mean(success_steps), 2),
                "rerun_successful_steps_std": round(statistics.stdev(success_steps), 2),
                "rerun_total_steps": total_steps,
                "rerun_total_overshoots": total_overshoots,
                "rerun_overshoot_rate": round(total_overshoots / total_steps * 100, 2),
                "successful_experiments": success_count,
                "total_experiments": len(rows),
            }
        ]
    )
    summary.to_csv(TABLES_DIR / "pid_rerun_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_experiment).to_csv(TABLES_DIR / "pid_rerun_per_experiment.csv", index=False, encoding="utf-8-sig")
    return summary


def calculate_acid_anion_charge(c_A: float, H: float, pKa_list: list[float]) -> float:
    n = len(pKa_list)
    K = [10 ** (-np.clip(pKa, -100, 100)) for pKa in pKa_list]
    denominator = 1.0
    cumulative_K = 1.0
    for i in range(n):
        cumulative_K *= K[i]
        denominator += cumulative_K / (H ** (i + 1))
    H_nA = c_A / denominator if denominator != 0 else 0.0
    anion_charge = 0.0
    cumulative_K = 1.0
    for i in range(n):
        cumulative_K *= K[i]
        anion_charge += (i + 1) * H_nA * (cumulative_K / (H ** (i + 1)))
    return anion_charge


def charge_balance(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa_list: list[float]) -> float:
    H = 10 ** (-pH)
    OH = 1e-14 / H
    return H + c_Na - OH - calculate_acid_anion_charge(c_A, H, pKa_list) - c_HCl


def solve_pH(c_A: float, c_Na: float, c_HCl: float, pKa_list: list[float]) -> float:
    lo, hi = 0.0, 14.0
    f_lo = charge_balance(lo, c_A, c_Na, c_HCl, pKa_list)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = charge_balance(mid, c_A, c_Na, c_HCl, pKa_list)
        if abs(f_mid) < 1e-10:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


def solve_volume_root(function, lo: float = 0.0, hi: float = 10.0, iterations: int = 80) -> float:
    f_lo = function(lo)
    f_hi = function(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return 0.0
    left, right = lo, hi
    left_value = f_lo
    for _ in range(iterations):
        mid = (left + right) / 2.0
        mid_value = function(mid)
        if abs(mid_value) < 1e-10:
            return mid
        if left_value * mid_value < 0:
            right = mid
        else:
            left = mid
            left_value = mid_value
    return (left + right) / 2.0


def calculate_acid_anion_charge_batch(c_A: float, H: np.ndarray, pKa_matrix: np.ndarray) -> np.ndarray:
    K = np.power(10.0, -np.clip(pKa_matrix, -100, 100))
    denominator = np.ones(H.shape[0], dtype=float)
    cumulative_K = np.ones(H.shape[0], dtype=float)
    for i in range(K.shape[1]):
        cumulative_K *= K[:, i]
        denominator += cumulative_K / np.power(H, i + 1)
    H_nA = c_A / denominator
    anion_charge = np.zeros(H.shape[0], dtype=float)
    cumulative_K = np.ones(H.shape[0], dtype=float)
    for i in range(K.shape[1]):
        cumulative_K *= K[:, i]
        anion_charge += (i + 1) * H_nA * (cumulative_K / np.power(H, i + 1))
    return anion_charge


def charge_balance_batch(pH: np.ndarray, c_A: float, c_Na: float, c_HCl: float, pKa_matrix: np.ndarray) -> np.ndarray:
    H = np.power(10.0, -pH)
    OH = 1e-14 / H
    acid_anion_charge = calculate_acid_anion_charge_batch(c_A, H, pKa_matrix)
    return H + c_Na - OH - acid_anion_charge - c_HCl


def solve_pH_batch(c_A: float, c_Na: float, c_HCl: float, pKa_matrix: np.ndarray) -> np.ndarray:
    n_particles = pKa_matrix.shape[0]
    lo = np.zeros(n_particles, dtype=float)
    hi = np.full(n_particles, 14.0, dtype=float)
    f_lo = charge_balance_batch(lo, c_A, c_Na, c_HCl, pKa_matrix)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = charge_balance_batch(mid, c_A, c_Na, c_HCl, pKa_matrix)
        left_mask = f_lo * f_mid < 0
        hi = np.where(left_mask, mid, hi)
        lo = np.where(left_mask, lo, mid)
        f_lo = np.where(left_mask, f_lo, f_mid)
    return (lo + hi) / 2.0


class BayesianParticleEnv:
    def __init__(self, num_particles: int = 1000) -> None:
        self.num_particles = num_particles
        self.reagents = {
            "Dilute acid 1": 0.1,
            "Dilute acid 2": 0.01,
            "Dilute base 1": 0.1,
            "Dilute base 2": 0.01,
        }
        self.min_addition_volume = 0.01
        self.addition_volumes = [round(self.min_addition_volume * i, 2) for i in range(1, 1000)]
        self.action_space = [(reagent, volume) for reagent in self.reagents for volume in self.addition_volumes]
        self.direction_penalty_factor = 60.0
        self.vol_ideal_factor = 0.2
        self.ph_rate_threshold = 1.0
        self.ph_rate_bonus_factor = 0.5
        self.max_steps = MAX_STEPS
        self.num_buffers = 3
        self.initialize_buffers()

    def initialize_buffers(self) -> None:
        self.pKa_list = np.random.uniform(2, 6, size=self.num_buffers)
        self.ref_pKa = np.copy(self.pKa_list)
        self.pKa_std = np.full(self.num_buffers, 0.2)
        self.buffer_total_moles = np.random.uniform(1e-6, 0.5, size=self.num_buffers)
        self.buffer_total_std = np.full(self.num_buffers, 0.005)

    def initialize(self, acid_type: str, acid_params: list[float], init_pH: float, target_pH: float, max_steps: int = MAX_STEPS) -> None:
        self.acid_type = acid_type
        self.true_pKas = acid_params if isinstance(acid_params, list) else [acid_params]
        self.current_ph = init_pH
        self.previous_ph = init_pH
        self.target_ph = target_pH
        self.steps_taken = 0
        self.done = False
        self.total_volume = INITIAL_ACID_VOL
        self.previous_total_volume = INITIAL_ACID_VOL
        self.acid_added_moles = 0.0
        self.base_added_moles = 0.0
        self.acid_volume = 0.0
        self.base_volume = 0.0
        self.last_acid_added = 0.0
        self.last_base_added = 0.0
        self.last_action_volume = 0.0
        self.last_measured_ph = init_pH
        self.prev_measured_ph = init_pH
        self.overshoot_threshold = None
        self.overshoot_occurred = False
        self.overshoot_reagent = None
        self.oscillation_count = 0
        self.use_secondary_reagents = False
        self.max_steps = max_steps
        self.initialize_buffers()

    def get_state(self) -> np.ndarray:
        previous = self.prev_measured_ph if self.prev_measured_ph is not None else self.current_ph
        pH_delta = self.current_ph - previous
        error = self.current_ph - self.target_ph
        return np.array([self.current_ph, self.target_ph, pH_delta, error, self.last_action_volume], dtype=np.float32)

    def update_exp_ph(self, pH: float) -> None:
        if self.last_measured_ph is not None:
            self.prev_measured_ph = self.last_measured_ph
        else:
            self.prev_measured_ph = pH
        self.current_ph = pH
        self.last_measured_ph = pH

    def get_effective_pka_array(self) -> np.ndarray:
        weight_max = 0.2
        weights = weight_max * (1 - np.tanh(self.pKa_std))
        return self.ref_pKa + weights * (self.pKa_list - self.ref_pKa)

    def get_effective_pka_matrix(self, sampled_pKa: np.ndarray) -> np.ndarray:
        weight_max = 0.2
        weights = weight_max * (1 - np.tanh(self.pKa_std))
        return self.ref_pKa + weights * (sampled_pKa - self.ref_pKa)

    def simulate_observed_ph(self) -> float:
        total_volume_l = (INITIAL_ACID_VOL + self.acid_volume + self.base_volume) / 1000.0
        n_analyte = (INITIAL_ACID_VOL / 1000.0) * 0.1
        c_A = n_analyte / total_volume_l
        c_Na = self.base_added_moles / total_volume_l
        c_HCl = self.acid_added_moles / total_volume_l
        return round(solve_pH(c_A, c_Na, c_HCl, self.true_pKas), 2)

    def compute_required_volume(self) -> float:
        n_analyte = (INITIAL_ACID_VOL / 1000.0) * 0.1
        effective_pKa = self.get_effective_pka_array().tolist()

        if self.current_ph < self.target_ph:
            reagent = "Dilute base 2" if self.use_secondary_reagents else "Dilute base 1"
            concentration = self.reagents[reagent]

            def function(volume: float) -> float:
                added_moles = concentration * (volume / 1000.0)
                new_base = self.base_added_moles + added_moles
                new_total_volume = (INITIAL_ACID_VOL + self.acid_volume + self.base_volume + volume) / 1000.0
                c_A_new = n_analyte / new_total_volume
                c_Na_new = new_base / new_total_volume
                c_HCl_new = self.acid_added_moles / new_total_volume
                return solve_pH(c_A_new, c_Na_new, c_HCl_new, effective_pKa) - self.target_ph

            return solve_volume_root(function, 0.0, 10.0)

        reagent = "Dilute acid 2" if self.use_secondary_reagents else "Dilute acid 1"
        concentration = self.reagents[reagent]

        def function(volume: float) -> float:
            added_moles = concentration * (volume / 1000.0)
            new_acid = self.acid_added_moles + added_moles
            new_total_volume = (INITIAL_ACID_VOL + self.acid_volume + self.base_volume + volume) / 1000.0
            c_A_new = n_analyte / new_total_volume
            c_Na_new = self.base_added_moles / new_total_volume
            c_HCl_new = new_acid / new_total_volume
            return solve_pH(c_A_new, c_Na_new, c_HCl_new, effective_pKa) - self.target_ph

        return solve_volume_root(function, 0.0, 10.0)

    def detect_overshoot(self, previous_ph: float, current_ph: float, reagent: str, last_added_moles: float) -> tuple[bool, float | None]:
        sign_change = (previous_ph - self.target_ph) * (current_ph - self.target_ph) < 0
        error_increased = abs(current_ph - self.target_ph) > abs(previous_ph - self.target_ph)
        if sign_change or error_increased:
            reagent_concentration = self.reagents[reagent]
            overshoot_volume = last_added_moles * 1000.0 / reagent_concentration
            return True, max(overshoot_volume / 2, self.min_addition_volume)
        return False, None

    def step(self, action: tuple[str, float], mode: str = "Simulate") -> tuple[float, float, bool, dict[str, bool]]:
        if self.done:
            return self.current_ph, 0.0, self.done, {}

        reagent, volume = action
        volume = float(volume)
        self.last_action_volume = volume
        added_moles = self.reagents[reagent] * (volume / 1000.0)
        self.previous_ph = self.current_ph
        self.previous_total_volume = self.total_volume
        self.total_volume += volume

        if "acid" in reagent.lower():
            self.acid_added_moles += added_moles
            self.acid_volume += volume
            self.last_acid_added = added_moles
        else:
            self.base_added_moles += added_moles
            self.base_volume += volume
            self.last_base_added = added_moles

        if mode != "Simulate":
            raise ValueError("This local benchmark supports only mode='Simulate'")

        simulated_ph = self.simulate_observed_ph()
        self.update_exp_ph(simulated_ph)

        if abs(volume - self.min_addition_volume) < 1e-6 and self.previous_ph is not None:
            if (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0 and abs(self.current_ph - self.previous_ph) > 0.1:
                self.oscillation_count += 1
                if self.oscillation_count >= 3:
                    self.use_secondary_reagents = True

        self.steps_taken += 1
        crossed_target = (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0
        last_added = self.last_acid_added if "acid" in reagent.lower() else self.last_base_added
        overshoot_flag, new_threshold = self.detect_overshoot(self.previous_ph, self.current_ph, reagent, last_added)
        if overshoot_flag:
            self.overshoot_occurred = True
            self.overshoot_reagent = reagent
            if new_threshold is not None and (self.overshoot_threshold is None or new_threshold < self.overshoot_threshold):
                self.overshoot_threshold = new_threshold

        current_error = abs(self.current_ph - self.target_ph)
        if current_error < SUCCESS_THRESHOLD or self.steps_taken >= self.max_steps:
            self.done = True

        reward = -current_error
        return self.current_ph, reward, self.done, {"crossed_target": crossed_target}

    def select_best_action(self) -> tuple[tuple[str, float], bool]:
        def filter_by_threshold(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
            if self.overshoot_threshold is not None:
                filtered = [candidate for candidate in candidates if candidate[1] <= self.overshoot_threshold]
                if filtered:
                    return filtered
            return candidates

        current_for_direction = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph

        if self.use_secondary_reagents:
            if self.overshoot_occurred and self.overshoot_reagent is not None:
                if "base" in self.overshoot_reagent.lower():
                    allowed = [name for name in self.reagents if "acid 2" in name.lower()]
                else:
                    allowed = [name for name in self.reagents if "base 2" in name.lower()]
            else:
                if current_for_direction < self.target_ph:
                    allowed = [name for name in self.reagents if "dilute base 2" in name.lower()]
                else:
                    allowed = [name for name in self.reagents if "dilute acid 2" in name.lower()]
        else:
            if self.overshoot_occurred and self.overshoot_reagent is not None:
                if "base" in self.overshoot_reagent.lower():
                    allowed = [name for name in self.reagents if "acid 1" in name.lower()]
                else:
                    allowed = [name for name in self.reagents if "base 1" in name.lower()]
                self.overshoot_occurred = False
                self.overshoot_reagent = None
            else:
                if current_for_direction < self.target_ph:
                    allowed = [name for name in self.reagents if "dilute base 1" in name.lower()]
                else:
                    allowed = [name for name in self.reagents if "dilute acid 1" in name.lower()]

        candidates = [action for action in self.action_space if action[0] in allowed]
        candidates = filter_by_threshold(candidates)

        error = abs(current_for_direction - self.target_ph)
        ph_change = abs(current_for_direction - (self.prev_measured_ph if self.prev_measured_ph is not None else current_for_direction))
        bonus_factor = 1 + self.ph_rate_bonus_factor * (1 - min(ph_change, self.ph_rate_threshold) / self.ph_rate_threshold)
        avg_uncertainty = np.mean(self.pKa_std)
        uncertainty_factor = 1 - 0.1 * min(avg_uncertainty / 1.0, 1)
        buffer_mean = np.mean(self.buffer_total_moles)
        buffering_factor = np.clip(1.0 + 0.1 * (buffer_mean - 0.5), 0.95, 1.05)
        alpha = self.vol_ideal_factor * bonus_factor * uncertainty_factor * buffering_factor
        required_vol = self.compute_required_volume()
        combined_value = error + 0.1 * required_vol
        max_vol = max(self.addition_volumes)
        ideal_volume = self.min_addition_volume + (max_vol - self.min_addition_volume) * np.tanh(alpha * combined_value)
        best_action = min(candidates, key=lambda candidate: abs(candidate[1] - ideal_volume))
        return best_action, self.done

    def update_posteriors(self, action: tuple[str, float], observed_ph: float) -> None:
        sampled_pKa = np.random.normal(self.pKa_list, self.pKa_std, size=(self.num_particles, self.num_buffers))
        sampled_total_moles = np.random.normal(self.buffer_total_moles, self.buffer_total_std, size=(self.num_particles, self.num_buffers))
        effective_pKa = self.get_effective_pka_matrix(sampled_pKa)

        total_volume_l = (INITIAL_ACID_VOL + self.acid_volume + self.base_volume) / 1000.0
        n_analyte = (INITIAL_ACID_VOL / 1000.0) * 0.1
        c_A = n_analyte / total_volume_l
        c_Na = self.base_added_moles / total_volume_l
        c_HCl = self.acid_added_moles / total_volume_l
        predicted_ph = solve_pH_batch(c_A, c_Na, c_HCl, effective_pKa)

        weights = np.exp(-0.5 * ((observed_ph - predicted_ph) / 0.01) ** 2)
        weights += 1e-12
        weights /= weights.sum()

        indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights)
        resampled_pKa = sampled_pKa[indices]
        resampled_total_moles = sampled_total_moles[indices]

        self.pKa_list = resampled_pKa.mean(axis=0)
        self.pKa_std = resampled_pKa.std(axis=0) + 1e-3
        self.buffer_total_moles = resampled_total_moles.mean(axis=0)
        self.buffer_total_std = resampled_total_moles.std(axis=0) + 1e-3


def load_experiment_conditions() -> list[dict[str, object]]:
    frame = pd.read_csv(EXPERIMENT_SUMMARY)
    conditions = []
    for _, row in frame.iterrows():
        conditions.append(
            {
                "experiment": int(row["Experiment"]),
                "acid_type": row["Acid_Type"],
                "acid_params": parse_pkas(row["Acid_Params"]),
                "initial_ph": float(row["Initial_pH"]),
                "target_ph": float(row["Target_pH"]),
            }
        )
    return conditions


def build_network_state(condition: dict[str, object]) -> np.ndarray:
    initial_ph = float(condition["initial_ph"])
    target_ph = float(condition["target_ph"])
    error = initial_ph - target_ph
    return np.array([initial_ph, target_ph, 0.0, error, 0.0], dtype=np.float64)


def numpy_mlp_latency(states: np.ndarray, iterations: int) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    w1 = rng.standard_normal((5, 256), dtype=np.float64)
    b1 = rng.standard_normal(256, dtype=np.float64)
    w2 = rng.standard_normal((256, 256), dtype=np.float64)
    b2 = rng.standard_normal(256, dtype=np.float64)
    w3 = rng.standard_normal((256, 1000), dtype=np.float64)
    b3 = rng.standard_normal(1000, dtype=np.float64)

    latencies_ms = []
    for index in range(iterations):
        x = states[index % len(states)]
        start = time.perf_counter_ns()
        h1 = np.maximum(0.0, np.dot(x, w1) + b1)
        h2 = np.maximum(0.0, np.dot(h1, w2) + b2)
        logits = np.dot(h2, w3) + b3
        _ = int(np.argmax(logits))
        end = time.perf_counter_ns()
        latencies_ms.append((end - start) / 1_000_000.0)
    return float(np.mean(latencies_ms)), float(np.median(latencies_ms))


def benchmark_latencies() -> pd.DataFrame:
    conditions = load_experiment_conditions()
    selected = conditions[:60]
    bayesian_latencies = []

    np.random.seed(555)
    for condition in selected:
        env = BayesianParticleEnv(num_particles=1000)
        env.initialize(condition["acid_type"], condition["acid_params"], condition["initial_ph"], condition["target_ph"])
        action, _ = env.select_best_action()
        current_ph, _, _, _ = env.step(action, mode="Simulate")
        start = time.perf_counter_ns()
        env.update_posteriors(action, current_ph)
        env.select_best_action()
        end = time.perf_counter_ns()
        bayesian_latencies.append((end - start) / 1_000_000.0)

    states = np.vstack([build_network_state(condition) for condition in selected])
    imitation_mean, imitation_median = numpy_mlp_latency(states, iterations=10000)
    reinforcement_mean, reinforcement_median = numpy_mlp_latency(states, iterations=10000)

    rows = [
        {
            "policy": "Bayesian update+select",
            "benchmark_type": "exact_local_numpy_port",
            "mean_latency_ms": round(float(np.mean(bayesian_latencies)), 4),
            "median_latency_ms": round(float(np.median(bayesian_latencies)), 4),
            "test_calls": len(bayesian_latencies),
            "comparison_to_20s_cycle_pct": round(float(np.mean(bayesian_latencies)) / 20000.0 * 100, 6),
            "notes": "Measures one posterior-update plus next-action-selection cycle using the local Bayesian logic ported from main_code3.ipynb cell 19.",
        },
        {
            "policy": "Imitation MLP forward pass",
            "benchmark_type": "architecture_only_approximation",
            "mean_latency_ms": round(imitation_mean, 4),
            "median_latency_ms": round(imitation_median, 4),
            "test_calls": 10000,
            "comparison_to_20s_cycle_pct": round(imitation_mean / 20000.0 * 100, 6),
            "notes": "Exact trained weights are not present locally and torch is unavailable in the active environment, so this is a numpy forward-pass approximation for the documented 5->256->256->1000 action head.",
        },
        {
            "policy": "Reinforcement MLP forward pass",
            "benchmark_type": "architecture_only_approximation",
            "mean_latency_ms": round(reinforcement_mean, 4),
            "median_latency_ms": round(reinforcement_median, 4),
            "test_calls": 10000,
            "comparison_to_20s_cycle_pct": round(reinforcement_mean / 20000.0 * 100, 6),
            "notes": "Same local architectural approximation as imitation learning because the RL policy reuses the same action head shape in code.",
        },
    ]

    frame = pd.DataFrame(rows)
    frame["python"] = sys.version.split()[0]
    frame["platform"] = platform.platform()
    frame.to_csv(BENCHMARK_DIR / "timing_benchmark_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def build_interpretability_support() -> tuple[pd.DataFrame, list[Path]]:
    conditions = load_experiment_conditions()
    representatives = {}
    for condition in conditions:
        acid_type = str(condition["acid_type"]).lower()
        if acid_type not in representatives:
            representatives[acid_type] = condition
        if len(representatives) == 3:
            break

    traces = []
    np.random.seed(777)
    for acid_type in ["monoprotic", "diprotic", "triprotic"]:
        condition = representatives[acid_type]
        env = BayesianParticleEnv(num_particles=1000)
        env.initialize(condition["acid_type"], condition["acid_params"], condition["initial_ph"], condition["target_ph"])
        traces.append(
            {
                "acid_type": acid_type,
                "step": 0,
                "mean_pKa_std": float(np.mean(env.pKa_std)),
                "max_pKa_std": float(np.max(env.pKa_std)),
                "current_pH": float(env.current_ph),
                "target_pH": float(env.target_ph),
                "experiment": int(condition["experiment"]),
            }
        )

        action, _ = env.select_best_action()
        while not env.done and env.steps_taken < 12:
            current_ph, _, _, _ = env.step(action, mode="Simulate")
            env.update_posteriors(action, current_ph)
            traces.append(
                {
                    "acid_type": acid_type,
                    "step": int(env.steps_taken),
                    "mean_pKa_std": float(np.mean(env.pKa_std)),
                    "max_pKa_std": float(np.max(env.pKa_std)),
                    "current_pH": float(env.current_ph),
                    "target_pH": float(env.target_ph),
                    "experiment": int(condition["experiment"]),
                }
            )
            if env.done:
                break
            action, _ = env.select_best_action()

    frame = pd.DataFrame(traces)
    frame.to_csv(TABLES_DIR / "bayesian_interpretability_uncertainty_trace.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {"monoprotic": "#457b9d", "diprotic": "#2a9d8f", "triprotic": "#e76f51"}
    for acid_type, subset in frame.groupby("acid_type"):
        ax.plot(
            subset["step"],
            subset["mean_pKa_std"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label="{} (exp {})".format(acid_type, int(subset["experiment"].iloc[0])),
            color=color_map[acid_type],
        )

    ax.set_xlabel("Bayesian update step")
    ax.set_ylabel("Mean posterior pKa std")
    ax.set_title("Representative posterior uncertainty shrinkage")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)

    png_path = FIGURES_DIR / "bayesian_uncertainty_shrinkage.png"
    svg_path = FIGURES_DIR / "bayesian_uncertainty_shrinkage.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for acid_type, subset in frame.groupby("acid_type"):
        initial_std = float(subset.loc[subset["step"] == 0, "mean_pKa_std"].iloc[0])
        final_std = float(subset["mean_pKa_std"].iloc[-1])
        summary_rows.append(
            {
                "acid_type": acid_type,
                "experiment": int(subset["experiment"].iloc[0]),
                "initial_mean_std": initial_std,
                "final_mean_std": final_std,
                "relative_drop_pct": round((initial_std - final_std) / initial_std * 100, 2),
            }
        )
    pd.DataFrame(summary_rows).to_csv(TABLES_DIR / "bayesian_interpretability_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    return frame, [png_path, svg_path]


def build_reproducibility_checklist() -> list[dict[str, str]]:
    return [
        {
            "Priority": "High",
            "Item": "Replace the template README with project-specific usage notes.",
            "Why it matters": "The current README does not explain the notebook structure, data layout, or how to rerun the simulation and physical plotting workflows.",
            "Evidence": str(REPO_ROOT / "README.md"),
        },
        {
            "Priority": "High",
            "Item": "Add a real environment file (`requirements.txt` or `environment.yml`).",
            "Why it matters": "The copied repo has no pinned environment spec, while the notebooks rely on numpy/pandas/matplotlib/scipy/torch/openpyxl.",
            "Evidence": str(MAIN_NOTEBOOK),
        },
        {
            "Priority": "High",
            "Item": "Split the 7000-line notebook into scripts or importable modules.",
            "Why it matters": "Single-notebook execution makes it hard to reproduce only one stage such as data generation, imitation training, RL fine-tuning, or PID benchmarking.",
            "Evidence": str(MAIN_NOTEBOOK),
        },
        {
            "Priority": "Medium",
            "Item": "Export trained weights and generated train/validation/test JSON files.",
            "Why it matters": "The local repo references `.pth` and dataset JSON files that are not present, which blocks exact timing and evaluation reruns for learned policies.",
            "Evidence": str(MAIN_NOTEBOOK),
        },
        {
            "Priority": "Medium",
            "Item": "Version the figure scripts separately from exploratory notebook cells.",
            "Why it matters": "The plotting notebook contains many overlapping figure variants, so it is not obvious which cell produced the manuscript figures.",
            "Evidence": str(PLOT_NOTEBOOK),
        },
    ]


def find_policy_usage_notes() -> list[dict[str, str]]:
    notes = [
        {
            "experiment": "Experiment 2 wastewater neutralization",
            "policy_in_text": "Reinforcement learning policy",
            "evidence": "{} (line containing 'The reinforcement learning policy then autonomously dosed')".format(SI_TEXT),
        },
        {
            "experiment": "Experiment 3 casein / milk",
            "policy_in_text": "Reinforcement learning policy",
            "evidence": "{} (line containing 'the reinforcement learning policy guided the sequential addition')".format(MANUSCRIPT_TEXT),
        },
        {
            "experiment": "Experiment 4 protein hydrolysis / later biochemical task",
            "policy_in_text": "Bayesian inference-guided controller",
            "evidence": "{} (line containing 'using our Bayesian inference-guided titration controller')".format(MANUSCRIPT_TEXT),
        },
    ]
    pd.DataFrame(notes).to_csv(TABLES_DIR / "policy_usage_audit.csv", index=False, encoding="utf-8-sig")
    return notes


def build_figure_audit_notes() -> list[dict[str, str]]:
    notes = [
        {
            "issue": "Confusing x-axis tick pattern",
            "finding": "plot.ipynb defines `_make_sparse_xticks` as [0, every 5 steps, final step], which directly explains labels like 5 and 8.",
            "evidence": "{}".format(PLOT_NOTEBOOK),
            "status": "fixed in draft candidate figure",
        },
        {
            "issue": "Inconsistent pH scales within rows",
            "finding": "Multiple plotting variants create subplots with `sharey=False`, so row-wise pH comparison is visually unstable.",
            "evidence": "{}".format(PLOT_NOTEBOOK),
            "status": "fixed in draft candidate figure",
        },
        {
            "issue": "Errant line artifact",
            "finding": "The notebook contains several overlapping plotting variants; a stray line can plausibly come from one of the line-segment versions. A clean replot without those segments is feasible, but the exact published cell remains uncertain.",
            "evidence": "{}".format(PLOT_NOTEBOOK),
            "status": "mitigated by clean draft candidate figure",
        },
        {
            "issue": "Figure 1 panel a black box",
            "finding": "No clear script or raw figure asset for the Figure 1 panel-a occlusion was identified in the copied repo.",
            "evidence": "{}; {}".format(PLOT_NOTEBOOK, MAIN_NOTEBOOK),
            "status": "manual source-figure audit still needed",
        },
    ]
    pd.DataFrame(notes).to_csv(TABLES_DIR / "figure_script_audit.csv", index=False, encoding="utf-8-sig")
    return notes


def build_prioritized_task_list(raw_metrics: dict[str, dict[str, float]], pid_rerun: pd.DataFrame) -> None:
    high_priority = [
        "Replace all manuscript, SI, and plotting-notebook simulation metrics with the trusted raw-log values from `data/bayesian.txt`, `data/m_network.txt`, `data/reinforced_network.txt`, and `data/PIDexperiment.txt`.",
        "Align the methods text with the local code: REINFORCE rather than PPO, 5->256->256->1000 rather than 64-unit hidden layers, and the documented action-volume ranges.",
        "Add the timing benchmark and explicitly compare decision latency against the 20 s mixing/sensing cycle.",
        "Weaken or qualify the strongest interpretability claim: the local code supports posterior updating and uncertainty shrinkage, but not robust identification of the true number of species.",
    ]
    medium_priority = [
        "Use the revised Figure 2 candidate and add the effective mixed-acid titration curves as SI support.",
        "Decide whether to include the adaptive PID baseline directly in the rebuttal/SI now that the local code reruns cleanly.",
        "Clarify the controller used in each physical experiment and resolve the mixed-acid Mixture 3 reinforcement-learning step-count mismatch.",
    ]
    low_priority = [
        "Prepare a direct supervised-on-simulation baseline if the reviewer response needs one beyond the current Bayesian-teacher imitation setup.",
        "Add a real README, environment file, and script/module entry points before any public code release.",
        "Manually verify the SI equation/symbol rendering in Word/PDF export, because the extracted text still shows several encoding-sensitive passages.",
    ]

    lines = [
        "# Prioritized Task List",
        "",
        "## High priority",
        "",
    ]
    lines.extend(["- {}".format(item) for item in high_priority])
    lines.extend(["", "## Medium priority", ""])
    lines.extend(["- {}".format(item) for item in medium_priority])
    lines.extend(["", "## Low priority", ""])
    lines.extend(["- {}".format(item) for item in low_priority])
    write_text(PRIORITY_PATH, "\n".join(lines) + "\n")


def format_metrics_row(algorithm: str, metrics: dict[str, float], source: str) -> dict[str, object]:
    return {
        "Algorithm": algorithm,
        "Success rate (%)": round(metrics["success_rate"], 2),
        "Successful steps (mean ± sd)": "{:.2f} ± {:.2f}".format(metrics["steps_mean"], metrics["steps_std"]),
        "Overshoot rate (%)": round(metrics["overshoot_rate"], 2),
        "Successful experiments": "{}/{}".format(metrics["success"], metrics["total"]),
        "Evidence": source,
    }


def build_report(
    reviewer_mapping: list[dict[str, object]],
    raw_metrics_frame: pd.DataFrame,
    raw_metrics: dict[str, dict[str, float]],
    metrics_audit: pd.DataFrame,
    method_audit: pd.DataFrame,
    volume_summary: pd.DataFrame,
    mixed_acid_mapping: pd.DataFrame,
    pid_rerun: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    interpretability_trace: pd.DataFrame,
    interpretability_figures: list[Path],
    figure2_paths: list[Path],
    titration_curve_paths: list[Path],
    policy_usage_notes: list[dict[str, str]],
    figure_audit_notes: list[dict[str, str]],
    reproducibility_checklist: list[dict[str, str]],
) -> None:
    trusted_metrics_table = markdown_table(
        [
            format_metrics_row("Bayesian", raw_metrics["Bayesian"], str(DATA_DIR / "bayesian.txt")),
            format_metrics_row("Imitation", raw_metrics["Imitation"], str(DATA_DIR / "m_network.txt")),
            format_metrics_row("Reinforcement", raw_metrics["Reinforcement"], str(DATA_DIR / "reinforced_network.txt")),
            format_metrics_row("PID", raw_metrics["PID"], str(DATA_DIR / "PIDexperiment.txt")),
        ],
        ["Algorithm", "Success rate (%)", "Successful steps (mean ± sd)", "Overshoot rate (%)", "Successful experiments", "Evidence"],
    )

    mismatch_rows = []
    for _, row in metrics_audit.sort_values(["source_group", "algorithm"]).iterrows():
        if row["status"] == "rounded_match":
            continue
        mismatch_rows.append(
            {
                "Source group": row["source_group"],
                "Algorithm": row["algorithm"],
                "Claimed success (%)": row["claimed_success_rate"],
                "Trusted success (%)": row["trusted_success_rate"],
                "Claimed steps": row["claimed_avg_successful_steps"],
                "Trusted steps": row["trusted_avg_successful_steps"],
                "Claimed overshoot (%)": row["claimed_overshoot_rate"],
                "Trusted overshoot (%)": row["trusted_overshoot_rate"],
            }
        )

    benchmark_rows = []
    for _, row in benchmark_frame.iterrows():
        benchmark_rows.append(
            {
                "Policy": row["policy"],
                "Mean latency (ms)": row["mean_latency_ms"],
                "Median latency (ms)": row["median_latency_ms"],
                "Calls": row["test_calls"],
                "Share of 20 s cycle (%)": row["comparison_to_20s_cycle_pct"],
                "Notes": row["notes"],
            }
        )

    uncertainty_summary = pd.read_csv(TABLES_DIR / "bayesian_interpretability_uncertainty_summary.csv")
    uncertainty_rows = []
    for _, row in uncertainty_summary.iterrows():
        uncertainty_rows.append(
            {
                "Acid type": row["acid_type"],
                "Experiment": int(row["experiment"]),
                "Initial mean std": round(float(row["initial_mean_std"]), 4),
                "Final mean std": round(float(row["final_mean_std"]), 4),
                "Relative drop (%)": round(float(row["relative_drop_pct"]), 2),
            }
        )

    volume_rows = []
    for _, row in volume_summary.iterrows():
        volume_rows.append(
            {
                "Category": row["category"],
                "Dataset": row["dataset"],
                "Min (mL)": row["min_mL"],
                "Max (mL)": row["max_mL"],
                "Resolution / min increment (mL)": row["resolution_mL"],
                "Evidence": row["files"],
            }
        )

    mixed_acid_issue = mixed_acid_mapping[
        (mixed_acid_mapping["mixture"] == 3) & (mixed_acid_mapping["controller"] == "Reinforcement")
    ].iloc[0]

    pid_summary_row = pid_rerun.iloc[0]
    pid_rerun_text = (
        "The extracted adaptive PID code reran cleanly against `experiment_summary.csv` and reproduced "
        "a success rate of {success:.2f}%, {steps:.2f} ± {std:.2f} successful steps, and {overshoot:.2f}% overshoot rate. "
        "Those numbers match the raw `data/PIDexperiment.txt` summary block."
    ).format(
        success=pid_summary_row["rerun_success_rate"],
        steps=pid_summary_row["rerun_avg_successful_steps"],
        std=pid_summary_row["rerun_successful_steps_std"],
        overshoot=pid_summary_row["rerun_overshoot_rate"],
    )

    summary_paragraph = (
        "This analysis stayed entirely inside copied materials. The main outcomes are: "
        "(1) the raw result logs support Bayesian / imitation / reinforcement success rates of 90.30 / 93.87 / 94.30%, "
        "not the stale ~94 / ~94 / ~94 values still hardcoded in the manuscript, SI, and plotting notebook; "
        "(2) the local RL implementation is REINFORCE with a 5->256->256->1000 policy head, not PPO with 64-unit layers; "
        "(3) the Bayesian controller is slower than the learned-policy architecture, but both are still negligible relative to the 20 s experimental cycle; "
        "and (4) the local Bayesian code exposes posterior-update quantities, but the current local traces do not support a strong quantitative interpretability claim."
    )

    lines = [
        "# Reviewer Analysis Report",
        "",
        "## Short summary",
        "",
        summary_paragraph,
        "",
        "## Copied working paths",
        "",
        "- Repo copy: `{}`".format(SOURCE_REPO_COPY),
        "- Manuscript copy: `{}`".format(SOURCE_MANUSCRIPT_COPY),
        "- SI copy: `{}`".format(SOURCE_SI_COPY),
        "- Reviewer PDF copy: `{}`".format(PDF_COPY),
        "",
        "## Reviewer-to-analysis mapping",
        "",
        markdown_table(reviewer_mapping, list(reviewer_mapping[0].keys())),
        "",
        "## Consistency audit findings",
        "",
        "### Trusted metrics summary",
        "",
        trusted_metrics_table,
        "",
        "The raw result logs are the most trustworthy local metric sources because they each contain a complete 3000-experiment summary block. "
        "The markdown/Word artifacts and plotting notebook contain multiple stale metric sets.",
        "",
        "### Mismatches against manuscript / SI / plotting notebook",
        "",
        markdown_table(
            mismatch_rows,
            [
                "Source group",
                "Algorithm",
                "Claimed success (%)",
                "Trusted success (%)",
                "Claimed steps",
                "Trusted steps",
                "Claimed overshoot (%)",
                "Trusted overshoot (%)",
            ],
        ),
        "",
        "### Additional method-description inconsistencies",
        "",
        markdown_table(
            method_audit.to_dict("records"),
            ["topic", "text_claim", "code_evidence", "local_files", "impact"],
        ),
        "",
        "### Physical benchmark file consistency",
        "",
        "The 12 mixed-acid files map cleanly to 4 mixtures x 3 controllers if suffix 1/2/3 is interpreted as Bayesian / reinforcement / human. "
        "That mapping reproduces the manuscript step counts for 11 of the 12 cells, but `{}` reports 12 reinforcement-learning steps for Mixture 3, whereas the manuscript narrative says 10.".format(
            MIXED_ACID_DIR / mixed_acid_issue["file"]
        ),
        "",
        markdown_table(
            mixed_acid_mapping.to_dict("records"),
            ["mixture", "controller", "file", "steps", "final_pH"],
        ),
        "",
        "## Timing benchmark results",
        "",
        markdown_table(
            benchmark_rows,
            ["Policy", "Mean latency (ms)", "Median latency (ms)", "Calls", "Share of 20 s cycle (%)", "Notes"],
        ),
        "",
        "Machine/environment notes: Python `{}` on `{}`. The learned-policy timing is an architecture-only approximation because the copied repo does not contain the `.pth` weight files or the exported dataset JSONs, and `torch` is not installed in the active Python 3.11 environment.".format(
            benchmark_frame["python"].iloc[0],
            benchmark_frame["platform"].iloc[0],
        ),
        "",
        "## PID / expert-rule baseline findings",
        "",
        pid_rerun_text,
        "",
        "Because the PID code exists as a standalone notebook cell with deterministic equations and cleanly reruns on `experiment_summary.csv`, it is mature enough to cite as a reviewer-facing classical baseline. "
        "Its main weakness is efficiency rather than overshoot rate: it succeeds much less often and uses many more steps than the learning-based policies.",
        "",
        "## Direct simulated-data baseline findings",
        "",
        "No distinct 'train a neural network directly on simulated data without a Bayesian teacher' baseline was found in the copied repo. "
        "The local imitation-learning pipeline uses simulated trajectories generated by the Bayesian controller itself (`main_code3.ipynb` cell 3), so it is still a teacher-distillation setup rather than a direct simulation-only supervised baseline.",
        "",
        "Minimal local experiment proposal: generate state-action labels by solving the simulator directly for the one-step volume that minimizes next-step absolute pH error, then train the same 5->256->256->1000 MLP on those labels and evaluate it on the same 3000-task benchmark.",
        "",
        "## Volume-range summary",
        "",
        markdown_table(
            volume_rows,
            ["Category", "Dataset", "Min (mL)", "Max (mL)", "Resolution / min increment (mL)", "Evidence"],
        ),
        "",
        "Key clarification: the Bayesian simulation code searches 0.01-9.99 mL over four reagents, whereas the learned policies discretize 0.01-10.00 mL. "
        "In the physical SSA and wastewater files, `recommended_volume` can exceed the executed `actual_volume` because early additions are capped at 5.0 mL and acid doses are rescaled by the neutralization factor.",
        "",
        "## SI titration-curve outputs",
        "",
        "Generated outputs:",
        "- `{}`".format(titration_curve_paths[0]),
        "- `{}`".format(titration_curve_paths[1]),
        "",
        "These are empirical / effective response curves for the four mixed-acid benchmark systems using cumulative signed titrant volume. "
        "I did not generate theoretical equilibrium curves because the copied local materials do not provide a single, authoritative table of the exact pKa values and mixture definitions needed for a defensible first-principles reconstruction.",
        "",
        "## Bayesian interpretability support findings",
        "",
        markdown_table(
            uncertainty_rows,
            ["Acid type", "Experiment", "Initial mean std", "Final mean std", "Relative drop (%)"],
        ),
        "",
        "Generated outputs:",
        "- `{}`".format(interpretability_figures[0]),
        "- `{}`".format(interpretability_figures[1]),
        "",
        "Interpretation: the local Bayesian routine does expose posterior means and standard deviations step-by-step, so interpretability is not purely rhetorical. "
        "However, the representative local traces above do not show clean monotonic uncertainty shrinkage; in two of the three examples the mean posterior pKa standard deviation actually increased over the sampled episode. "
        "That means the current local materials are better used to support a weak claim about visible internal Bayesian state than a strong claim about reliably inferring the true chemical composition or pKa values.",
        "",
        "## Figure-script audit findings",
        "",
        markdown_table(
            figure_audit_notes,
            ["issue", "finding", "evidence", "status"],
        ),
        "",
        "Draft revised figure outputs:",
        "- `{}`".format(figure2_paths[0]),
        "- `{}`".format(figure2_paths[1]),
        "",
        "## Policy-usage clarification",
        "",
        markdown_table(policy_usage_notes, ["experiment", "policy_in_text", "evidence"]),
        "",
        "## Reproducibility checklist",
        "",
        markdown_table(reproducibility_checklist, ["Priority", "Item", "Why it matters", "Evidence"]),
        "",
        "## Open gaps / unresolved issues",
        "",
        "- Exact learned-policy latency cannot be reproduced from the copied repo because the `.pth` model weights referenced by the notebook are absent and `torch` is unavailable in the active Python environment.",
        "- The manuscript, SI, and plotting notebook currently mix at least two incompatible simulation-metric sets; manuscript writing should pick one trusted source before any further revision text is drafted.",
        "- Figure 1 panel-a source assets were not identifiable from the copied repo, so the black-box occlusion issue still needs a manual figure-source audit.",
        "- The extracted text still shows multiple encoding-sensitive equation/symbol passages in the SI, so comment 15 should be addressed by a manual Word/PDF export check rather than by this code-only audit.",
        "",
        "## Prioritized task list",
        "",
        "See `{}` for the concise High / Medium / Low priority version.".format(PRIORITY_PATH),
        "",
        "## Generated files",
        "",
        "- Tables: `{}`".format(TABLES_DIR),
        "- Benchmarks: `{}`".format(BENCHMARK_DIR),
        "- Figures: `{}`".format(FIGURES_DIR),
        "- Logs / extracted cells: `{}`".format(LOGS_DIR),
    ]
    write_text(REPORT_PATH, "\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    extract_text_artifacts()

    maybe_export_cell(MAIN_NOTEBOOK, 1, LOGS_DIR / "main_code3_cell_1.py")
    maybe_export_cell(MAIN_NOTEBOOK, 19, LOGS_DIR / "main_code3_cell_19.py")
    maybe_export_cell(MAIN_NOTEBOOK, 33, LOGS_DIR / "main_code3_cell_33.py")

    reviewer_mapping = build_reviewer_mapping()
    pd.DataFrame(reviewer_mapping).to_csv(TABLES_DIR / "reviewer_mapping.csv", index=False, encoding="utf-8-sig")

    raw_metrics_frame, raw_metrics = parse_simulation_metrics()
    metrics_audit = build_metrics_claim_audit(raw_metrics)
    method_audit = build_method_consistency_audit()
    volume_summary, _ = summarize_volume_ranges()
    mixed_acid_mapping = summarize_mixed_acid_controller_mapping()
    pid_rerun = rerun_pid_summary()
    benchmark_frame = benchmark_latencies()
    interpretability_trace, interpretability_figures = build_interpretability_support()
    figure2_paths = plot_figure2_candidate()
    titration_curve_paths, _ = plot_effective_titration_curves()
    policy_usage_notes = find_policy_usage_notes()
    figure_audit_notes = build_figure_audit_notes()
    reproducibility_checklist = build_reproducibility_checklist()
    build_prioritized_task_list(raw_metrics, pid_rerun)

    build_report(
        reviewer_mapping=reviewer_mapping,
        raw_metrics_frame=raw_metrics_frame,
        raw_metrics=raw_metrics,
        metrics_audit=metrics_audit,
        method_audit=method_audit,
        volume_summary=volume_summary,
        mixed_acid_mapping=mixed_acid_mapping,
        pid_rerun=pid_rerun,
        benchmark_frame=benchmark_frame,
        interpretability_trace=interpretability_trace,
        interpretability_figures=interpretability_figures,
        figure2_paths=figure2_paths,
        titration_curve_paths=titration_curve_paths,
        policy_usage_notes=policy_usage_notes,
        figure_audit_notes=figure_audit_notes,
        reproducibility_checklist=reproducibility_checklist,
    )

    print("Report written to {}".format(REPORT_PATH))
    print("Priority list written to {}".format(PRIORITY_PATH))


if __name__ == "__main__":
    main()
