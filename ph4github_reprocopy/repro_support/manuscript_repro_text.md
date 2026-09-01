# Draft Reproducibility Text for Manuscript / SI

The manuscript and SI working copies were resolved and copied during this pass:

- Manuscript copy: `Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\manuscript\Hybrid Bayesian Inference and Reinforcement Learning for Autonomous pH Adjustment in Diverse Chemical Systems2.0_reprocopy.docx`
- SI copy: `Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\supplementary_information\SI-V19_reprocopy.docx`

This memo does not edit those documents directly. It provides text blocks that can be inserted into the manuscript revision and SI where appropriate.

## Code availability

Code used for the computational analyses is provided in the accompanying repository. For the revision, the repository package was supplemented with a practical `README`, dependency notes, and lightweight helper scripts under `repro_support/` that allow third parties to run basic checks, generate a representative bundled figure, and execute the revised PID baseline on the released `experiment_summary.csv` file.

## Data availability

The repository includes the bundled experiment summary table (`experiment_summary.csv`), archived text outputs for several model and baseline runs (`data/*.txt`), and released experimental tables under `data/all_data/`. Some original plotting notebook cells reference additional local files outside the repository; those gaps are documented explicitly in `repro_support/reproducibility_report.md`. A repository-specific data and code availability statement should therefore distinguish between files included in the public package and files that remained local during figure preparation.

## Software environment note

Because the original development environment was not preserved as a fully pinned environment file, the released package documents a practical tested setup rather than an exact historical reconstruction. The repository now includes `requirements.txt` and `reproducibility_notes.md`, which distinguish smoke-tested dependencies from notebook-only dependencies that may require manual installation for broader reruns.

## Limitations statement

The released package preserves the original working notebooks for transparency, but they remain partially monolithic and are not yet a fully modular pipeline. To improve practical reproducibility for the revision, a minimal standalone PID baseline script and relative-path plotting/check scripts were added. Full end-to-end reruns of every notebook section may still require additional dependency installation and further cleanup of machine-specific plotting paths.

## Reviewer-response style sentence

To address the reviewer's concerns about repository usability, we replaced the placeholder documentation with a project-specific README, added dependency and execution notes, documented hard-coded path limitations, and extracted minimal runnable scripts for repository checks, representative figure generation, and the PID baseline.

## Observed wording gaps in the current draft set

- The manuscript copy contains a `Data availability` heading that still reads like a placeholder prompt rather than a final article-specific statement.
- The SI already states that simulation randomness was controlled with `seed = 555` and already points readers to the GitHub repository, so the main missing pieces are the environment note and the clearer description of what is directly rerunnable from the release package.
