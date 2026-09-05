"""Independently recalculate successes from the new task-level output."""
from pathlib import Path
import csv,json,statistics,math
WORK=Path(__file__).resolve().parent
OUT=WORK/'full_5x5x3000'
SEEDS=(101,202,303,404,555)
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
expected={101:(91.68,0.41),202:(90.46,0.64),303:(93.95,0.63),404:(92.59,0.57),555:(90.29,0.43)}
recorded=read(OUT/'per_training_seed_summary.csv')
recomputed=[];total=0
for seed in SEEDS:
    rates=[]
    for benchmark in SEEDS:
        rows=read(OUT/f'ppo_{seed}_benchmark_{benchmark}.csv')
        assert len(rows)==3000
        assert len({(r['task_seed'],r['task_id']) for r in rows})==3000
        for r in rows:
            assert int(r['training_seed'])==seed and int(r['benchmark_seed'])==benchmark
            error=abs(float(r['final_true_ph'])-float(r['target_ph']))
            assert math.isclose(error,float(r['final_abs_error']),abs_tol=1e-12)
            assert int(r['true_success'])==int(error<=0.10)
            assert 0<=int(r['steps'])<=50
        rates.append(100*sum(int(r['true_success']) for r in rows)/3000)
        total+=len(rows)
    avg=statistics.mean(rates);sd=statistics.stdev(rates)
    reference=next(r for r in recorded if int(r['training_seed'])==seed)
    assert math.isclose(avg,float(reference['success_rate_percent_mean']),abs_tol=1e-12)
    assert math.isclose(sd,float(reference['success_rate_percent_sd']),abs_tol=1e-12)
    matches=(round(avg,2),round(sd,2))==expected[seed]
    recomputed.append({'training_seed':seed,'benchmark_success_percent':rates,'mean':avg,'sample_sd':sd,'matches_previously_reported_two_decimal_values':matches})
complete=json.loads((OUT/'COMPLETE.json').read_text())
assert sum(r['tasks_compared'] for r in complete['selected_model_reference_checks'])==15000
assert sum(r['mismatched_fields'] for r in complete['selected_model_reference_checks'])==0
overall=statistics.mean(r['mean'] for r in recomputed)
spread=statistics.stdev(r['mean'] for r in recomputed)
checks={'status':'PASS','raw_rows_rechecked':total,'training_seed_summary':recomputed,'overall_mean':overall,'overall_sample_sd':spread,'overall_matches_previous':(round(overall,2),round(spread,2))==(91.79,1.53),'selected_303_reference_mismatches':0,'ppo_higher_than_imitation_cells':complete['ppo_higher_than_imitation_cells']}
(OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
lines=['# Primary PPO re-evaluation — 6 September 2026','',
       'All five saved PPO policies were evaluated on the unchanged five locked sets of 3,000 tasks: 75,000 task-level evaluations, without retraining or checkpoint selection. Generated results are separate from the publication evidence.','',
       'The archived primary evaluator was used unchanged, including its 0.10-pH success threshold, 0.01-pH observation rounding, 50-addition/50-mL cumulative limits, direction rule and persistent post-overshoot dose cap. All five task-manifest hashes and model-file hashes passed preflight.','',
       '| Training seed | Success mean (%) | Sample SD across benchmark sets (%) | Matches previous two-decimal report |','|---|---:|---:|---|']
for row in recomputed:
    lines.append(f"| {row['training_seed']} | {row['mean']:.8f} | {row['sample_sd']:.8f} | {row['matches_previously_reported_two_decimal_values']} |")
lines+=['',f'Across the five training-seed means: **{overall:.8f} ± {spread:.8f}%**. This reproduces the previously reported **91.79 ± 1.53%**, or **92 ± 2%** under the manuscript’s one-significant-digit SD convention. These extra digits are retained here for audit purposes, not proposed as manuscript formatting.','',
        f"PPO success was higher than the corresponding archived imitation success in **{complete['ppo_higher_than_imitation_cells']}/25** training-seed × benchmark-set comparisons. This comparison concerns numerical success rates, not a claim that all differences are statistically significant.",'',
        'For the selected seed-303 model, all 15,000 rows matched the archived primary results on the shared output fields (numerical tolerance 1e-12). An independent pass recomputed every success indicator from final true pH and target pH, checked task counts and identities, and recalculated the mean and sample SD.','',
        'Outputs: `tmp/primary_ppo_recheck_20260906/full_5x5x3000/`, including 25 task-level CSVs, per-cell/per-training-seed summaries, comparisons against imitation, run provenance, and `INDEPENDENT_AUDIT.json`.','',
        '## Local-window posterior RMSE code','',
        'A general implementation exists in `13_SOURCE_ARCHIVES/joint_parameter_bayesian_code_current/experiment_utils.py`, function `curve_metrics` (line 167), invoked by `run_pf_curve_recovery.py`. It compares predicted and true pH changes after subtracting each curve’s value at zero additional titrant. Default windows are ±0.10, ±0.25 and ±0.50 mL, sampled on a 0.01-mL grid.','',
        'This is code for the separate joint-parameter study; it has not been established as the script producing the SI values 0.0399, 0.1280 and 0.2452 for ±0.10, ±0.50 and ±1.00 mL.','',
        'The supplied S6 package’s `docs/ANALYSIS_AND_DISCUSSION_CN.md` (lines 6, 24–26) explicitly records those three SI values and says that both curves were anchored at the current pH, but the original local-window script and task-level outputs were not found. Thus, a code building block and a written result record exist; the exact numerical source remains unresolved. No replacement local-window analysis or document edits were made.']
report=WORK.parents[1]/'exports/Primary_PPO_recheck_and_local_RMSE_code_20260906.md'
report.write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(checks,indent=2))
print(report)
