"""Recalculate the archived Table S6 / Response Table R7 study without rerunning PF."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / 'evidence/simulation_numerical_evidence_20260823/09_PF_RULE_ABLATIONS/internal_rule_reproduction_20260905'
SEEDS = (101, 202, 303, 404, 555)
VARIANTS = ('full', 'no_ph_rate_bonus', 'no_uncertainty_factor', 'no_buffering_factor', 'no_required_volume_term', 'linear_clip_instead_of_tanh')
METRICS = ('success_rate_percent', 'successful_steps_mean', 'crossings_mean', 'total_volume_mean_ml', 'final_abs_error_mean')

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def rounded_pair(mean, sd):
    m, s = Decimal(str(mean)), Decimal(str(sd))
    rs = s.quantize(Decimal(1).scaleb(s.adjusted()), rounding=ROUND_HALF_UP)
    unit = Decimal(1).scaleb(rs.adjusted())
    return f'{m.quantize(unit, rounding=ROUND_HALF_UP):f} ± {rs.normalize():f}'

def assert_close(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=2e-11, abs_tol=1e-12):
        raise ValueError(f'{label}: {a} != {b}')

def holm(rows):
    running = 0.0
    for rank, i in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]['raw_p'])):
        running = max(running, min(1.0, (len(rows)-rank)*rows[i]['raw_p']))
        rows[i]['holm_adjusted_p'] = running

def audit(block=BLOCK, paired_continuous=False):
    block = Path(block)
    hashed = 0
    for line in (block/'SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        expected, relative = line.split('  ', 1)
        path = (block/relative).resolve()
        if not path.is_relative_to(block.resolve()):
            raise ValueError('Hash manifest path escapes the archive')
        data = path.read_bytes()
        if expected not in {hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()}:
            raise ValueError(f'Content hash mismatch: {relative}')
        hashed += 1
    result_dir = block/'results/formal_results'
    text_fields = {'variant', 'direction', 'difficulty'}
    integer_fields = {'task_id','task_seed','true_pair_count','success','strict_success','severe_failure','steps','crossings','benchmark_seed'}
    rows = read_csv(result_dir/'all_task_results.csv')
    for r in rows:
        for k in r:
            if k not in text_fields:
                r[k] = int(r[k]) if k in integer_fields else float(r[k])
        if not all(math.isfinite(v) for v in r.values() if isinstance(v,float)):
            raise ValueError('Non-finite task result')
        assert r['success'] == int(r['final_abs_error'] <= .1)
        assert r['strict_success'] == int(r['final_abs_error'] <= .05)
        assert r['severe_failure'] == int(r['final_abs_error'] > .5)
        assert 0 <= r['crossings'] <= r['steps'] <= 50
    counts = Counter((r['benchmark_seed'],r['variant']) for r in rows)
    assert counts == Counter({(s,v):300 for s in SEEDS for v in VARIANTS})
    lookup = {(r['benchmark_seed'],r['task_id'],r['variant']):r for r in rows}
    assert len(lookup) == len(rows) == 9000
    tasks = {}
    for seed in SEEDS:
        records = [json.loads(line) for line in (result_dir/f'seed_{seed}_tasks.jsonl').read_text().splitlines() if line.strip()]
        assert len(records) == 300
        for task in records:
            assert (seed,task['task_id']) not in tasks
            tasks[(seed,task['task_id'])] = task
    for r in rows:
        task = tasks[(r['benchmark_seed'],r['task_id'])]
        assert task['seed'] == r['task_seed'] == 5000000+r['benchmark_seed']
        assert (r['direction'],r['difficulty'],r['true_pair_count']) == (task['direction'],task['difficulty'],len(task['pka_values']))
    groups = defaultdict(list)
    for r in rows:
        groups[(r['benchmark_seed'],r['variant'])].append(r)
    per_seed = {}
    for key, group in groups.items():
        successful = [r for r in group if r['success']]
        per_seed[key] = {
            'tasks':len(group),
            'success_rate_percent':100*st.mean(r['success'] for r in group),
            'strict_success_rate_percent':100*st.mean(r['strict_success'] for r in group),
            'severe_failure_rate_percent':100*st.mean(r['severe_failure'] for r in group),
            'successful_steps_mean':st.mean(r['steps'] for r in successful),
            'crossings_mean':st.mean(r['crossings'] for r in group),
            'total_volume_mean_ml':st.mean(r['total_volume_ml'] for r in group),
            'final_abs_error_mean':st.mean(r['final_abs_error'] for r in group),
            'controller_ms_per_step_mean':st.mean(r['controller_ms_per_step'] for r in group)}
    for row in read_csv(result_dir/'per_seed_summary.csv'):
        for key,value in per_seed[(int(row['benchmark_seed']),row['variant'])].items():
            assert_close(row[key],value,f'Per-seed {key}')
    aggregate = {}
    for variant in VARIANTS:
        summary = {}
        for metric in per_seed[(101,variant)]:
            values = [per_seed[(s,variant)][metric] for s in SEEDS]
            summary[metric+'_mean'] = st.mean(values)
            summary[metric+'_sd'] = st.stdev(values)
        aggregate[variant] = summary
    for row in read_csv(result_dir/'aggregate_summary.csv'):
        for key,value in aggregate[row['variant']].items():
            assert_close(row[key],value,f'Aggregate {key}')
    success_tests, continuous_tests = [], []
    keys = sorted(tasks)
    for variant in VARIANTS[1:]:
        full = [lookup[(s,i,'full')] for s,i in keys]
        alt = [lookup[(s,i,variant)] for s,i in keys]
        a = sum(x['success'] and not y['success'] for x,y in zip(full,alt))
        b = sum(y['success'] and not x['success'] for x,y in zip(full,alt))
        n = a+b
        p = min(1.0, 2*sum(math.comb(n,k) for k in range(min(a,b)+1))/2**n) if n else 1.0
        success_tests.append({'comparison':variant+'_minus_full','paired_tasks':1500,'full_only_success':a,'variant_only_success':b,'success_difference_pp':100*(b-a)/1500,'raw_p':p})
        if paired_continuous:
            from scipy.stats import wilcoxon
            for metric in ('steps','crossings','total_volume_ml','final_abs_error'):
                x = [r[metric] for r in full]; y = [r[metric] for r in alt]
                differences = [b-a for a,b in zip(x,y)]
                continuous_tests.append({'comparison':variant+'_minus_full','metric':metric,'paired_tasks':1500,'mean_difference':st.mean(differences),'median_difference':st.median(differences),'raw_p':float(wilcoxon(y,x,zero_method='zsplit').pvalue)})
    holm(success_tests)
    if paired_continuous:
        holm(continuous_tests)
    for filename, checked in [('paired_success_tests.csv',success_tests),('paired_continuous_tests.csv',continuous_tests)]:
        if not checked:
            continue
        archived = read_csv(result_dir/filename)
        assert len(archived) == len(checked)
        for old,new in zip(archived,checked):
            for key,value in new.items():
                if isinstance(value,str):
                    assert old[key] == value
                else:
                    assert_close(old[key],value,f'{filename}: {key}')
    return {'status':'PASS','archive_files_verified':hashed,'unique_tasks':len(tasks),'task_results':len(rows),
            'seed_variant_groups':len(groups),'exact_McNemar_Holm_tests':len(success_tests),
            'Wilcoxon_Holm_tests':len(continuous_tests),
            'table_s6':{v:{m:rounded_pair(aggregate[v][m+'_mean'],aggregate[v][m+'_sd']) for m in METRICS} for v in VARIANTS},
            'successful_additions_differences':{v:aggregate[v]['successful_steps_mean_mean']-aggregate['full']['successful_steps_mean_mean'] for v in VARIANTS[1:]}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-continuous',action='store_true',help='Also recompute all Wilcoxon/Holm tests; requires SciPy.')
    args=parser.parse_args()
    print(json.dumps(audit(paired_continuous=args.paired_continuous),indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()
