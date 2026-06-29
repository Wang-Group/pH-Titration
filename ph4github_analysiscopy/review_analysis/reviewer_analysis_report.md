# Reviewer Analysis Report

## Short summary

This analysis stayed entirely inside copied materials. The main outcomes are: (1) the raw result logs support Bayesian / imitation / reinforcement success rates of 90.30 / 93.87 / 94.30%, not the stale ~94 / ~94 / ~94 values still hardcoded in the manuscript, SI, and plotting notebook; (2) the local RL implementation is REINFORCE with a 5->256->256->1000 policy head, not PPO with 64-unit layers; (3) the Bayesian controller is slower than the learned-policy architecture, but both are still negligible relative to the 20 s experimental cycle; and (4) the local Bayesian code exposes posterior-update quantities, but the current local traces do not support a strong quantitative interpretability claim.

## Copied working paths

- Repo copy: `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy`
- Manuscript copy: `Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\manuscript\Hybrid Bayesian Inference and Reinforcement Learning for Autonomous pH Adjustment in Diverse Chemical Systems2.0_analysiscopy.docx`
- SI copy: `Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\supplementary_information\SI-V19_analysiscopy.docx`
- Reviewer PDF copy: `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\pdf_text\SC-EDG-05-2026-003882_Round1_analysiscopy.pdf`

## Reviewer-to-analysis mapping

| Reviewer comment ID | Short summary | Why it needs analysis/code | Relevant local files | Proposed deliverable |
| --- | --- | --- | --- | --- |
| 1a | Clarify what is novel beyond prior RL/Bayesian pH-control work. | Requires direct method comparison and exact local evidence for Bayesian vs IL vs RL behavior. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\bayesian.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\m_network.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\reinforced_network.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt | Trusted simulation-metrics table plus method-consistency audit. |
| 1b | Substantiate interpretability claims for Bayesian control. | Needs posterior/uncertainty traces from the local Bayesian update code. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\experiment_summary.csv | Posterior-uncertainty shrinkage figure and limitations note. |
| 1c | Explain Bayesian policy advantage over a neural network trained directly on simulation. | Requires inspecting how the imitation dataset is generated and whether a direct simulation-only baseline exists. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\si_analysiscopy.txt | Baseline-status note plus a minimal direct-supervised experiment proposal. |
| 3 | Benchmark computational cost against the 20 s experimental cycle. | Requires timing measurements on the local Bayesian code path and the learned-policy architecture. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\experiment_summary.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\si_analysiscopy.txt | Timing benchmark CSV and manuscript-ready comparison paragraph. |
| 4 | Assess whether the human advantage in Figure 2 is just small-sample noise. | Requires reconstructing the physical benchmark mapping and counting the true sample size. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\si_analysiscopy.txt | Physical-step audit with an explicit n=4 limitation statement. |
| 4a | Evaluate an expert-rule or PID-style baseline. | Requires locating, running, and summarizing the baseline code. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\experiment_summary.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\PIDexperiment.txt | Recomputed PID summary and comparison table. |
| 6a | Fix the Figure 2 x-axis labeling pattern. | Requires auditing notebook plotting logic and producing a revised draft figure script. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid | Revised Figure 2 candidate with step ticks every 2 steps plus final step. |
| 6b | Remove the errant line in Figure 2. | Needs a clean plotting implementation with controlled line/marker rendering. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid | Revised Figure 2 candidate without stray connecting artifacts. |
| 6c | Use consistent pH axes within each row of Figure 2. | Needs a row-shared y-limit plotting pass. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid | Revised Figure 2 candidate with row-consistent pH limits. |
| 7 | Provide titration-curve plots for the example buffered systems. | Needs SI-ready figure generation from local physical datasets or local models. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\si_analysiscopy.txt | Effective empirical titration-curve figure and an explicit note on the lack of exact theoretical inputs. |
| 8a | Replace unquantified 'significantly' language with numbers. | Needs exact effect sizes from the trusted metrics audit. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\bayesian.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\m_network.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\reinforced_network.txt | Percentage-point and step-count deltas ready for manuscript wording. |
| 9 | Clarify what the physical dosing increments/resolution really were. | Needs exact action-volume ranges from code and physical logs. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data | Simulation-vs-physical volume-range summary table. |
| 10 | Clarify which policy was used in Experiments 3 and 4. | Needs a manuscript/SI policy-usage audit against the local text and datasets. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\si_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data | Policy-usage note with exact local text evidence. |
| 11 | Report input volume ranges for training and experiments. | Needs extraction from notebook code and physical experiment files. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data | Action-space summary table with dataset-specific min/max values. |
| 12 | Single giant notebook hinders reproducibility. | Needs a practical reproducibility checklist tied to the current repository state. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\README.md | Revision-ready reproducibility checklist. |
| 13 | The repository README and environment documentation are incomplete. | Needs an actionable list of missing documentation and environment files. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\README.md; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb | README/requirements/notebook-extraction checklist. |

## Consistency audit findings

### Trusted metrics summary

| Algorithm | Success rate (%) | Successful steps (mean ± sd) | Overshoot rate (%) | Successful experiments | Evidence |
| --- | --- | --- | --- | --- | --- |
| Bayesian | 90.3 | 8.96 ± 5.92 | 44.3 | 2709/3000 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\bayesian.txt |
| Imitation | 93.87 | 10.26 ± 7.21 | 31.25 | 2816/3000 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\m_network.txt |
| Reinforcement | 94.3 | 10.22 ± 7.24 | 31.41 | 2829/3000 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\reinforced_network.txt |
| PID | 76.17 | 23.70 ± 10.82 | 29.88 | 2285/3000 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\PIDexperiment.txt |

The raw result logs are the most trustworthy local metric sources because they each contain a complete 3000-experiment summary block. The markdown/Word artifacts and plotting notebook contain multiple stale metric sets.

### Mismatches against manuscript / SI / plotting notebook

| Source group | Algorithm | Claimed success (%) | Trusted success (%) | Claimed steps | Trusted steps | Claimed overshoot (%) | Trusted overshoot (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| manuscript_table1_and_narrative | Bayesian | 94.2 | 90.3 | 12.73 | 8.96 | 41.84 | 44.3 |
| manuscript_table1_and_narrative | Imitation | 93.77 | 93.87 | 10.22 | 10.26 | 34.41 | 31.25 |
| manuscript_table1_and_narrative | Reinforcement | 94.27 | 94.3 | 10.21 | 10.22 | 30.55 | 31.41 |
| plot_notebook_hardcoded | Bayesian | 94.23 | 90.3 | 12.73 | 8.96 | 41.84 | 44.3 |
| plot_notebook_hardcoded | Imitation | 93.77 | 93.87 | 10.22 | 10.26 | 34.41 | 31.25 |
| plot_notebook_hardcoded | Reinforcement | 94.27 | 94.3 | 10.21 | 10.22 | 30.55 | 31.41 |
| si_stale_paragraph | Bayesian | 94.2 | 90.3 | 13.73 | 8.96 | 39.99 | 44.3 |
| si_stale_paragraph | Imitation | 93.97 | 93.87 | 10.2 | 10.26 | 28.44 | 31.25 |
| si_stale_paragraph | PID | 29.703 | 76.17 | 44.19 | 23.7 | 24.4 | 29.88 |
| si_stale_paragraph | Reinforcement | 94.327 | 94.3 | 10.1 | 10.22 | 28.67 | 31.41 |

### Additional method-description inconsistencies

| topic | text_claim | code_evidence | local_files | impact |
| --- | --- | --- | --- | --- |
| imitation_architecture | Manuscript says the imitation MLP has two hidden layers of 64 units and is trained with MSE. | main_code3.ipynb cells 5 and 7 define a 5->256->256->1000 classifier trained with CrossEntropyLoss. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_5.py | Method description is inconsistent with the local implementation. |
| reinforcement_algorithm | Manuscript says PPO with 50,000 episodes; SI says REINFORCE but also contains 500/1000-episode text. | main_code3.ipynb cell 9 implements REINFORCE and trains for 500 episodes by default. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\si_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_9.py | The RL method section needs a full consistency pass before revision. |
| imitation_dataset_source | Text sometimes frames imitation as simulation-only data without distinguishing the Bayesian teacher. | main_code3.ipynb cell 3 generates state-action pairs by calling the Bayesian controller (select_best_action and update_posteriors). | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_3.py | The direct-simulated-data baseline requested by the reviewer does not already exist as a distinct local model. |
| volume_range | Text gives imprecise or mixed volume-range language. | Bayesian controller uses 0.01-9.99 mL candidate volumes; learned policies use 0.01-10.00 mL discrete bins; PID caps at 3.00 mL. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_1.py; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_5.py; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_33.py | The methods section should separate Bayesian simulation, learned-policy action bins, and physical dosing limits. |

### Physical benchmark file consistency

The 12 mixed-acid files map cleanly to 4 mixtures x 3 controllers if suffix 1/2/3 is interpreted as Bayesian / reinforcement / human. That mapping reproduces the manuscript step counts for 11 of the 12 cells, but `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\3-2.xlsx` reports 12 reinforcement-learning steps for Mixture 3, whereas the manuscript narrative says 10.

| mixture | controller | file | steps | final_pH |
| --- | --- | --- | --- | --- |
| 1 | Bayesian | 1-1.xlsx | 12 | 6.05 |
| 1 | Reinforcement | 1-2.xlsx | 9 | 5.98 |
| 1 | Human | 1-3.xlsx | 4 | 6.05 |
| 2 | Bayesian | 2-1.xlsx | 2 | 6.06 |
| 2 | Reinforcement | 2-2.xlsx | 3 | 5.98 |
| 2 | Human | 2-3.xlsx | 8 | 5.96 |
| 3 | Bayesian | 3-1.xlsx | 7 | 5.93 |
| 3 | Reinforcement | 3-2.xlsx | 12 | 5.9 |
| 3 | Human | 3-3.xlsx | 5 | 5.98 |
| 4 | Bayesian | 4-1.xlsx | 8 | 6.06 |
| 4 | Reinforcement | 4-2.xlsx | 8 | 6.02 |
| 4 | Human | 4-3.xlsx | 6 | 5.99 |

## Timing benchmark results

| Policy | Mean latency (ms) | Median latency (ms) | Calls | Share of 20 s cycle (%) | Notes |
| --- | --- | --- | --- | --- | --- |
| Bayesian update+select | 53.5692 | 32.8419 | 60 | 0.267846 | Measures one posterior-update plus next-action-selection cycle using the local Bayesian logic ported from main_code3.ipynb cell 19. |
| Imitation MLP forward pass | 0.166 | 0.126 | 10000 | 0.00083 | Exact trained weights are not present locally and torch is unavailable in the active environment, so this is a numpy forward-pass approximation for the documented 5->256->256->1000 action head. |
| Reinforcement MLP forward pass | 0.2111 | 0.1356 | 10000 | 0.001055 | Same local architectural approximation as imitation learning because the RL policy reuses the same action head shape in code. |

Machine/environment notes: Python `3.11.9` on `Windows-10-10.0.26200-SP0`. The learned-policy timing is an architecture-only approximation because the copied repo does not contain the `.pth` weight files or the exported dataset JSONs, and `torch` is not installed in the active Python 3.11 environment.

## PID / expert-rule baseline findings

The extracted adaptive PID code reran cleanly against `experiment_summary.csv` and reproduced a success rate of 76.17%, 23.70 ± 10.82 successful steps, and 29.88% overshoot rate. Those numbers match the raw `data/PIDexperiment.txt` summary block.

Because the PID code exists as a standalone notebook cell with deterministic equations and cleanly reruns on `experiment_summary.csv`, it is mature enough to cite as a reviewer-facing classical baseline. Its main weakness is efficiency rather than overshoot rate: it succeeds much less often and uses many more steps than the learning-based policies.

## Direct simulated-data baseline findings

No distinct 'train a neural network directly on simulated data without a Bayesian teacher' baseline was found in the copied repo. The local imitation-learning pipeline uses simulated trajectories generated by the Bayesian controller itself (`main_code3.ipynb` cell 3), so it is still a teacher-distillation setup rather than a direct simulation-only supervised baseline.

Minimal local experiment proposal: generate state-action labels by solving the simulator directly for the one-step volume that minimizes next-step absolute pH error, then train the same 5->256->256->1000 MLP on those labels and evaluate it on the same 3000-task benchmark.

## Volume-range summary

| Category | Dataset | Min (mL) | Max (mL) | Resolution / min increment (mL) | Evidence |
| --- | --- | --- | --- | --- | --- |
| simulation | Bayesian controller action space | 0.01 | 9.99 | 0.01 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_1.py |
| simulation | Imitation policy action bins | 0.01 | 10.0 | 0.01 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_5.py |
| simulation | Reinforcement policy action bins | 0.01 | 10.0 | 0.01 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_9.py |
| simulation | Adaptive PID baseline output | 0.01 | 3.0 | 0.001 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs\main_code3_cell_33.py |
| physical | mixed_acid | 0.06 | 7.869999885559082 | 7.993605777301127e-15 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\1-1.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\1-2.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\1-3.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\2-1.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\2-2.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\2-3.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\3-1.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\3-2.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\3-3.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\4-1.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\4-2.xlsx; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\mixed_acid\4-3.xlsx |
| physical | milk | 0.43 | 5.21 | 0.06000000000000005 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\milk\milk.xlsx |
| physical | ssa | 0.05 | 5.0 | 0.0003570153461996961 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA1.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA10.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA11.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA2.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA3.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA4.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA5.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA6.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA7.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA8.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\SSA\SSA9.csv |
| physical | wastewater | 0.05 | 5.0 | 0.0003570153461996961 | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\wastewater\WasteWater1.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\wastewater\WasteWater2.csv; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\data\all_data\wastewater\WasteWater3.csv |

Key clarification: the Bayesian simulation code searches 0.01-9.99 mL over four reagents, whereas the learned policies discretize 0.01-10.00 mL. In the physical SSA and wastewater files, `recommended_volume` can exceed the executed `actual_volume` because early additions are capped at 5.0 mL and acid doses are rescaled by the neutralization factor.

## SI titration-curve outputs

Generated outputs:
- `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\figures\effective_titration_curves_mixed_acid.png`
- `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\figures\effective_titration_curves_mixed_acid.svg`

These are empirical / effective response curves for the four mixed-acid benchmark systems using cumulative signed titrant volume. I did not generate theoretical equilibrium curves because the copied local materials do not provide a single, authoritative table of the exact pKa values and mixture definitions needed for a defensible first-principles reconstruction.

## Bayesian interpretability support findings

| Acid type | Experiment | Initial mean std | Final mean std | Relative drop (%) |
| --- | --- | --- | --- | --- |
| diprotic | 8 | 0.2 | 0.212 | -5.98 |
| monoprotic | 2 | 0.2 | 0.2222 | -11.11 |
| triprotic | 1 | 0.2 | 0.2063 | -3.13 |

Generated outputs:
- `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\figures\bayesian_uncertainty_shrinkage.png`
- `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\figures\bayesian_uncertainty_shrinkage.svg`

Interpretation: the local Bayesian routine does expose posterior means and standard deviations step-by-step, so interpretability is not purely rhetorical. However, the representative local traces above do not show clean monotonic uncertainty shrinkage; in two of the three examples the mean posterior pKa standard deviation actually increased over the sampled episode. That means the current local materials are better used to support a weak claim about visible internal Bayesian state than a strong claim about reliably inferring the true chemical composition or pKa values.

## Figure-script audit findings

| issue | finding | evidence | status |
| --- | --- | --- | --- |
| Confusing x-axis tick pattern | plot.ipynb defines `_make_sparse_xticks` as [0, every 5 steps, final step], which directly explains labels like 5 and 8. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb | fixed in draft candidate figure |
| Inconsistent pH scales within rows | Multiple plotting variants create subplots with `sharey=False`, so row-wise pH comparison is visually unstable. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb | fixed in draft candidate figure |
| Errant line artifact | The notebook contains several overlapping plotting variants; a stray line can plausibly come from one of the line-segment versions. A clean replot without those segments is feasible, but the exact published cell remains uncertain. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb | mitigated by clean draft candidate figure |
| Figure 1 panel a black box | No clear script or raw figure asset for the Figure 1 panel-a occlusion was identified in the copied repo. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb; E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb | manual source-figure audit still needed |

Draft revised figure outputs:
- `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\figures\figure2_candidate_revised.png`
- `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\figures\figure2_candidate_revised.svg`

## Policy-usage clarification

| experiment | policy_in_text | evidence |
| --- | --- | --- |
| Experiment 2 wastewater neutralization | Reinforcement learning policy | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\si_analysiscopy.txt (line containing 'The reinforcement learning policy then autonomously dosed') |
| Experiment 3 casein / milk | Reinforcement learning policy | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt (line containing 'the reinforcement learning policy guided the sequential addition') |
| Experiment 4 protein hydrolysis / later biochemical task | Bayesian inference-guided controller | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\manuscript_text\manuscript_analysiscopy.txt (line containing 'using our Bayesian inference-guided titration controller') |

## Reproducibility checklist

| Priority | Item | Why it matters | Evidence |
| --- | --- | --- | --- |
| High | Replace the template README with project-specific usage notes. | The current README does not explain the notebook structure, data layout, or how to rerun the simulation and physical plotting workflows. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\README.md |
| High | Add a real environment file (`requirements.txt` or `environment.yml`). | The copied repo has no pinned environment spec, while the notebooks rely on numpy/pandas/matplotlib/scipy/torch/openpyxl. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb |
| High | Split the 7000-line notebook into scripts or importable modules. | Single-notebook execution makes it hard to reproduce only one stage such as data generation, imitation training, RL fine-tuning, or PID benchmarking. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb |
| Medium | Export trained weights and generated train/validation/test JSON files. | The local repo references `.pth` and dataset JSON files that are not present, which blocks exact timing and evaluation reruns for learned policies. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\main_code3.ipynb |
| Medium | Version the figure scripts separately from exploratory notebook cells. | The plotting notebook contains many overlapping figure variants, so it is not obvious which cell produced the manuscript figures. | E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\plot.ipynb |

## Open gaps / unresolved issues

- Exact learned-policy latency cannot be reproduced from the copied repo because the `.pth` model weights referenced by the notebook are absent and `torch` is unavailable in the active Python environment.
- The manuscript, SI, and plotting notebook currently mix at least two incompatible simulation-metric sets; manuscript writing should pick one trusted source before any further revision text is drafted.
- Figure 1 panel-a source assets were not identifiable from the copied repo, so the black-box occlusion issue still needs a manual figure-source audit.
- The extracted text still shows multiple encoding-sensitive equation/symbol passages in the SI, so comment 15 should be addressed by a manual Word/PDF export check rather than by this code-only audit.

## Prioritized task list

See `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_analysis\prioritized_task_list.md` for the concise High / Medium / Low priority version.

## Generated files

- Tables: `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\tables`
- Benchmarks: `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\benchmarks`
- Figures: `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\figures`
- Logs / extracted cells: `E:\GitHub\ph4git\pH-Titration\ph4github_analysiscopy\review_outputs\logs`
