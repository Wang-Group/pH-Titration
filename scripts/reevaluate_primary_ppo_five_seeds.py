"""Re-evaluate all five saved primary PPO actors, without any model training.

Uses the unchanged archived primary evaluator, not the deployment wrapper.
Run from any working directory; --output must be a new directory.
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import argparse,csv,json,sys,time,hashlib,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.audit_primary_ppo_five_seeds import verify_inputs, canonical_bytes
EVIDENCE=REPO/'evidence/simulation_numerical_evidence_20260823'
SOURCE=EVIDENCE/'13_SOURCE_ARCHIVES/primary_locked_benchmark_source'
FORMAL=EVIDENCE/'01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation'
CHECKPOINTS=EVIDENCE/'02_TEACHER_AND_IMITATION/checkpoints'
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
import run_matched_evaluation as evaluator
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
SEEDS=(101,202,303,404,555)
EXPECTED={
101:'9fedcafa84f4d3f8969bfae486a200a4ab8147d6af4e9a427f8c41e998b0caf3',
202:'b4e9864280f5902456cb0e4dacfdd24c7ee2fa1c4df0a28b2377273319f9df0c',
303:'4004d7a09768fc5ac3f448523f53cb22210ed919ca7e713f13d9aa693cc19de5',
404:'d7aef675e54121a7cb88fb6b7225b457d6609b2c142ed29cab641d34cf6a8264',
555:'098b65d844ad0b06f0a960520cea7e672739d9abcd30b2f1e931920561af3ec7'}

def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def preflight():
    verify_inputs()
    manifests={}
    for seed in SEEDS:
        p=FORMAL/'tasks'/f'seed_{seed}_tasks.jsonl'
        actual=hashlib.sha256(canonical_bytes(p)).hexdigest()
        tasks=evaluator.load_tasks(p)
        if len(tasks)!=3000 or len({(t.seed,t.task_id) for t in tasks})!=3000:
            raise ValueError(f'Task manifest is incomplete: {p.name}')
        manifests[p.name]=actual
    weights={}
    for seed in SEEDS:
        p=CHECKPOINTS/f'principal_ppo_seed_{seed}.pth'
        actual=evaluator.sha256(p)
        if actual!=EXPECTED[seed]:
            raise ValueError(f'Checkpoint hash mismatch: {p.name}')
        weights[p.name]=actual
    return {'task_manifests':manifests,'checkpoints':weights,
            'input_hash_convention':'lf_normalized_text',
            'source_hashes':{p.relative_to(REPO).as_posix():hashlib.sha256(canonical_bytes(p)).hexdigest() for p in list(SOURCE.glob('*.py'))+[REPO/'controllers/controller_api.py']},
            'source_hash_convention':'SHA-256 after CRLF-to-LF normalization; checkpoints use exact bytes',
            'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'success_tolerance_ph':0.10,'max_additions':50,'max_total_dose_ml':50,
            'observation_rounding_ph':0.01,'action_range_ml':[0.01,10.0],
            'persistent_overshoot_cap':True,'evaluation':'argmax, no retraining',
            'training_seeds':list(SEEDS),'benchmark_seeds':list(SEEDS),'tasks_per_cell':3000}

def compare_selected(rows):
    reference=evaluator.read_csv(FORMAL/'all_task_results.csv')
    lookup={(int(r['benchmark_seed']),int(r['task_seed']),int(r['task_id'])):r for r in reference if r['method']=='ppo'}
    differences=[]
    for row in rows:
        ref=lookup[(row['benchmark_seed'],row['task_seed'],row['task_id'])]
        for key,value in row.items():
            if key not in ref or key in ('training_seed',):continue
            if isinstance(value,(int,float)):
                equal=abs(float(ref[key])-value)<=1e-12
            else:equal=str(value)==ref[key]
            if not equal: differences.append({'task':row['task_id'],'benchmark_seed':row['benchmark_seed'],'metric':key,'new':value,'reference':ref[key]})
    return {'tasks_compared':len(rows),'mismatched_fields':len(differences),'examples':differences[:20]}

def evaluate_cell(job):
    training_seed,benchmark_seed,count,output=job
    actor,normalizer,_=evaluator.load_checkpoint(CHECKPOINTS/f'principal_ppo_seed_{training_seed}.pth',torch.device('cpu'))
    tasks=evaluator.load_tasks(FORMAL/'tasks'/f'seed_{benchmark_seed}_tasks.jsonl')[:count]
    rows=[]
    for index,task in enumerate(tasks,1):
        row=evaluator.rollout_network(actor,normalizer,task,torch.device('cpu'),int(task.seed*1_000_003+task.task_id),use_overshoot_cap=True)
        row.update(method='ppo',benchmark_seed=benchmark_seed,training_seed=training_seed)
        rows.append(row)
        if count==3000 and index%1000==0:
            print(f'PPO {training_seed}, benchmark {benchmark_seed}: {index}/{count}',flush=True)
    path=Path(output)/f'ppo_{training_seed}_benchmark_{benchmark_seed}.csv'
    evaluator.write_csv(path,rows)
    result=evaluator.summarize(rows,'ppo',benchmark_seed)
    result['training_seed']=training_seed
    if training_seed==303:result['selected_model_reference_check']=compare_selected(rows)
    dump(path.with_suffix('.json'),result)
    return result

def aggregate(summaries,out,count):
    metric_names=evaluator.METRICS+('cap_activation_rate_percent',)
    five=[]
    for seed in SEEDS:
        subset=[r for r in summaries if r['training_seed']==seed]
        if len(subset)!=5:
            raise ValueError(f'Incomplete evaluation for seed {seed}')
        row={'training_seed':seed,'benchmark_sets':5,'evaluations':5*count}
        for metric in metric_names:
            values=np.array([r[metric] for r in subset])
            row[metric+'_mean']=float(values.mean())
            row[metric+'_sd']=float(values.std(ddof=1))
        five.append(row)
    evaluator.write_csv(out/'per_training_seed_summary.csv',five)
    means=np.array([r['success_rate_percent_mean'] for r in five])
    reference=evaluator.read_csv(FORMAL/'all_task_results.csv')
    imitation={}
    for seed in SEEDS:
        task_ids={t.task_id for t in evaluator.load_tasks(FORMAL/'tasks'/f'seed_{seed}_tasks.jsonl')[:count]}
        matched=[r for r in reference if r['method']=='imitation' and int(r['benchmark_seed'])==seed and int(r['task_id']) in task_ids]
        if len(matched)!=count:
            raise ValueError(f'Incomplete imitation reference for benchmark {seed}')
        imitation[seed]=100.0*float(np.mean([int(r['true_success']) for r in matched]))
    comparisons=[{'training_seed':r['training_seed'],'benchmark_seed':r['benchmark_seed'],'ppo_success_percent':r['success_rate_percent'],'imitation_success_percent':imitation[r['benchmark_seed']],'difference_pp':r['success_rate_percent']-imitation[r['benchmark_seed']]} for r in summaries]
    evaluator.write_csv(out/'paired_set_success_vs_imitation.csv',comparisons)
    result={'status':'COMPLETE','evaluations':25*count,'training_seed_success_mean_percent':float(means.mean()),'training_seed_success_sd_percent':float(means.std(ddof=1)),
            'training_seed_success_range_percent':[float(means.min()),float(means.max())],
            'ppo_higher_than_imitation_cells':sum(r['difference_pp']>0 for r in comparisons),'cells':25,
            'selected_model_reference_checks':[r['selected_model_reference_check'] for r in summaries if r['training_seed']==303]}
    dump(out/'COMPLETE.json',result)
    print(json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New output directory; existing directories are never overwritten')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--tasks-per-set',type=int,default=3000,help='Default 3000; smaller counts are technical smoke tests, not benchmark results')
    args=parser.parse_args()
    if args.workers<1 or not 1<=args.tasks_per_set<=3000:
        parser.error('workers must be positive; tasks-per-set must be in 1..3000')
    config=preflight()
    out=args.output.resolve()
    if out.is_relative_to(EVIDENCE.resolve()):
        parser.error('Write new runs outside the released evidence directory')
    out.mkdir(parents=True,exist_ok=False)
    config['workers']=args.workers
    config['phase']='full' if args.tasks_per_set==3000 else 'smoke'
    config['tasks_per_cell']=args.tasks_per_set
    dump(out/'RUN_CONFIG.json',config)
    jobs=[(a,b,args.tasks_per_set,str(out)) for a in SEEDS for b in SEEDS]
    summaries=[]
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(evaluate_cell,j) for j in jobs]):
            result=future.result()
            summaries.append(result)
            print(f"Completed {len(summaries)}/{len(jobs)} cells: PPO {result['training_seed']}, benchmark {result['benchmark_seed']}, success {result['success_rate_percent']:.6f}%; elapsed {time.monotonic()-start:.1f}s",flush=True)
            dump(out/'PROGRESS.json',{'completed_cells':len(summaries),'total_cells':len(jobs),'elapsed_s':time.monotonic()-start})
    summaries.sort(key=lambda r:(r['training_seed'],r['benchmark_seed']))
    evaluator.write_csv(out/'per_cell_summary.csv',[{k:v for k,v in r.items() if k!='selected_model_reference_check'} for r in summaries])
    aggregate(summaries,out,args.tasks_per_set)
    from scripts.audit_primary_ppo_five_seeds import audit as audit_results
    report=audit_results(out)
    dump(out/'REPRODUCTION_AUDIT.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
