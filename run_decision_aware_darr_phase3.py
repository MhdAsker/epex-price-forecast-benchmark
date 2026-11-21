"""Frozen, resumable 2016-2017 evaluation for Project A Prompt 3."""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from src.research.decision_aware_darr.causal import window_id
from src.research.decision_aware_darr.dataset import context_pattern_matrix, direct_prediction_lookup
from src.research.decision_aware_darr.embeddings import embedding_ids, extract_embeddings, load_embedding_store
from src.research.decision_aware_darr.forecast import heuristic_scores, retrieval_forecasts
from src.research.decision_aware_darr.phase3 import candidate_pairs, minimal_library, query_table
from src.research.decision_aware_darr.reranker import build_inference_pair_table, build_pair_table, load_checkpoint, predict
from src.research.decision_aware_darr.storage import atomic_json, atomic_npz, atomic_parquet, canonical_hash, output_record, sha256_file, verify_batch_manifest, verify_output
from src.research.kumo.data import build_origins, training_standardizer, validate_canonical_hourly
from src.research.kumo.inference import load_head
from src.research.kumo.train import build_verified_model

ROOT=Path(__file__).resolve().parent; CONFIG=ROOT/'config/decision_aware_darr_phase3.json'; RESULTS=ROOT/'results/decision_aware_darr/phase3'; ART=ROOT/'artifacts/decision_aware_darr/phase3'; PROGRESS=ROOT/'docs/DECISION_AWARE_DARR_PHASE3_PROGRESS.md'
P1=ROOT/'results/decision_aware_darr/phase1'; P2=ROOT/'results/decision_aware_darr/phase2'

def cfg():
    c=json.loads(CONFIG.read_text()); return c,canonical_hash(c)

def frozen_inputs():
    c,h=cfg(); p2m=json.loads((ROOT/c['inputs']['phase2_manifest']).read_text()); bad=[]
    for r in p2m['outputs']:
        ok,why=verify_output(r,ROOT)
        if not ok: bad.append((r['path'],why))
    freeze=json.loads((ROOT/c['inputs']['phase2_freeze']).read_text()); models={}
    for key in c['models']['frozen_pipelines']:
        eid=freeze['pipelines'][key]['experiment_id']; m=json.loads((P2/'experiments'/f'{eid}.json').read_text()); rec=next(x for x in m['outputs'] if x['path'].endswith('model.joblib')); ok,why=verify_output(rec,ROOT)
        if not ok: bad.append((rec['path'],why))
        models[key]={'selection':freeze['pipelines'][key],'record':rec,'experiment_id':eid}
    retro=json.loads((ROOT/c['inputs']['retrospective_manifest']).read_text()); pred_path=ROOT/c['inputs']['retrospective_predictions']; rec=next(x for x in retro['artifacts'] if Path(x['path'])==pred_path.relative_to(ROOT)); ok,why=verify_output(rec,ROOT)
    if not ok: bad.append((rec['path'],why))
    if bad: raise RuntimeError(f'frozen input failures: {bad}')
    return c,h,models,sha256_file(ROOT/c['inputs']['phase2_manifest'])

def origins():
    c,_,_,_=frozen_inputs(); frame=pd.read_parquet(ROOT/c['inputs']['source_data']); memory=build_origins(frame,target_start=c['memory']['candidate_start'],target_end_exclusive=c['evaluation']['end_exclusive']); queries=build_origins(frame,target_start=c['evaluation']['start'],target_end_exclusive=c['evaluation']['end_exclusive'],require_full_target=False); new=build_origins(frame,target_start='2015-12-30',target_end_exclusive=c['evaluation']['end_exclusive'],require_full_target=False)
    if len(queries)!=731: raise ValueError(len(queries))
    return frame,memory,queries,new

def reproduce_development(models):
    pair0=pd.read_parquet(P1/'candidate_features.parquet'); q=pd.read_parquet(P1/'query_features.parquet'); lib=pd.read_parquet(P1/'candidate_library.parquet'); table=build_inference_pair_table(pair0,q,lib); val=table[table['partition']=='validation']; qv=q[q.partition=='validation']; archived=pd.read_parquet(P2/'validation_forecasts.parquet'); metrics=pd.read_csv(P2/'validation_metrics.csv').set_index('method'); names={'ridge_forecast':'Ridge Forecast frozen','ridge_economic':'Ridge Economic frozen','lightgbm_forecast':'Lightgbm Forecast frozen','lightgbm_economic':'Lightgbm Economic frozen'}; rows=[]
    for key,item in models.items():
        ck=load_checkpoint(ROOT/item['record']['path']); scored=val[['query__id','candidate__id']].copy(); scored['score']=predict(ck['model'],ck['experiment'],val); s=item['selection']; forecasts=retrieval_forecasts(scored,lib,qv,score_column='score',k=s['k'],direct_weight=s['direct_weight']); name=names[key]; ref=archived[archived.method==name]; lookup={(r.query__id,int(r.hour)):float(r.prediction) for r in ref.itertuples()}; diffs=[abs(float(v[h])-lookup[(qid,h)]) for qid,v in forecasts.items() for h in range(24)]; rows.append({'pipeline':key,'max_abs_prediction_difference':max(diffs),'registered_MAE':float(metrics.loc[name,'MAE']),'registered_battery_revenue_eur':float(metrics.loc[name,'battery_revenue_eur']),'checkpoint_sha256':item['record']['sha256']})
    direct=float(metrics.loc['Direct Kumo','MAE']); report={'status':'passed' if max(x['max_abs_prediction_difference'] for x in rows)<1e-8 and abs(direct-6.287069831802769)<1e-9 else 'failed','tolerance':1e-8,'direct_kumo_MAE':direct,'pipelines':rows}
    if report['status']!='passed': raise RuntimeError(report)
    atomic_json(RESULTS/'development_reproduction.json',report); return report

def write_progress(stage,status='IN PROGRESS',extra=None):
    c,h,models,p2h=frozen_inputs(); extra=extra or {}; state={'status':status.lower(),'stage':stage,'config_sha256':h,'phase2_manifest_sha256':p2h,'embedding_batches_complete':extra.get('embedding_batches_complete',0),'query_batches_complete':extra.get('query_batches_complete',0),'query_batches_total':25,'retrospective_used_for_fitting':False,'next_command':extra.get('next_command','python run_decision_aware_darr_phase3.py status')}; atomic_json(RESULTS/'progress_state.json',state)
    hashes='\n'.join(f"  - `{k}`: `{v['record']['sha256']}`" for k,v in models.items()); PROGRESS.write_text(f"""# Decision-aware DARR Phase 3 progress

## Current state

- Status: {status}.
- Branch: `feature/decision-aware-darr-phase3`.
- Parent commit: `93ca62c`.
- Last verified stage: {stage}.
- Phase 2 manifest SHA-256: `{p2h}`.
- Frozen checkpoint hashes:\n{hashes}
- Development reproduction: {'passed' if (RESULTS/'development_reproduction.json').exists() else 'pending'}.
- Embedding batches complete: {state['embedding_batches_complete']} / 12.
- Query batches complete: {state['query_batches_complete']} / 25.
- 2016–2017 used for fitting/tuning: no.
- Kumo/rerankers retrained: no.
- Tests: {json.loads((RESULTS/'test_status.json').read_text())['summary'] if (RESULTS/'test_status.json').exists() else 'pending'}.

## Next command
```powershell
{state['next_command']}
```
""",encoding='utf-8')

def run_plan():
    c,h,models,p2h=frozen_inputs(); frame,memory,queries,new=origins(); repro=reproduce_development(models); ordered=validate_canonical_hourly(frame); training=ordered[(ordered.index>='2012-02-01')&(ordered.index<'2015-01-01')].price_eur_mwh; daily=training.groupby(training.index.normalize()).std(); plan={'status':'registered','config_sha256':h,'phase2_manifest_sha256':p2h,'development_reproduction':repro['status'],'queries':len(queries),'first_date':str(queries[0].delivery_date.date()),'last_date':str(queries[-1].delivery_date.date()),'candidate_memory_windows':len(memory),'new_embedding_windows':len(new),'memory_policy':c['memory'],'high_volatility_daily_std_threshold':float(daily.quantile(.75)),'spike_threshold':100.0,'retrospective_not_pristine':True,'spread_control':c['spread_amplification_control']}; atomic_json(RESULTS/'plan.json',plan); write_progress('Stage 3 protocol registered',extra={'next_command':'python run_decision_aware_darr_phase3.py build --resume'}); print(json.dumps(plan,indent=2))

def embed_status(c,h,new):
    total=math.ceil(len(new)/c['batching']['embedding_windows']); out=[]
    for n in range(total): out.append(verify_batch_manifest(RESULTS/'batches/embeddings'/f'embedding_{n:03d}.json',ROOT,h,sha256_file(ROOT/c['inputs']['source_data']))[0])
    return out

def query_status(c,h):
    source=sha256_file(ROOT/c['inputs']['source_data']); return [verify_batch_manifest(RESULTS/'batches/queries'/f'query_{n:03d}.json',ROOT,h,source)[0] for n in range(25)]

def all_embeddings(c,h,frame,memory,queries,new):
    p1c=json.loads((ROOT/c['inputs']['phase1_config']).read_text()); old=load_embedding_store(root=ROOT,manifest_dir=P1/'batches/embeddings',expected_ids=embedding_ids(memory[:1428]),config_hash=canonical_hash(p1c),source_hash=p1c['hashes']['source_dataset_sha256']); store=dict(zip(embedding_ids(memory[:1428]),old)); source=sha256_file(ROOT/c['inputs']['source_data'])
    for path in sorted((RESULTS/'batches/embeddings').glob('embedding_*.json')):
        ok,why=verify_batch_manifest(path,ROOT,h,source)
        if not ok: raise RuntimeError((path,why))
        m=json.loads(path.read_text()); data=np.load(ROOT/m['outputs'][0]['path']); store.update(dict(zip(data['ids'].astype(str),data['embeddings'].astype(np.float32))))
    required=set(embedding_ids(memory))|set(embedding_ids(queries)); missing=required-set(store)
    if missing: raise RuntimeError(f'missing embeddings {len(missing)}')
    return store

def run_build(resume):
    c,h,models,_=frozen_inputs(); frame,memory,queries,new=origins(); source=sha256_file(ROOT/c['inputs']['source_data']); es=embed_status(c,h,new)
    if not all(es):
        kumo=json.loads((ROOT/'config/kumo_finetuning.json').read_text()); p1=json.loads((ROOT/c['inputs']['phase1_config']).read_text()); model,report=build_verified_model(kumo,'cpu'); load_head(model,ROOT/p1['model']['head_checkpoint']); std=training_standardizer(frame,train_start='2012-02-01',train_end_exclusive='2015-01-01'); size=c['batching']['embedding_windows']
        for n,done in enumerate(es):
            if done: continue
            batch=new[n*size:(n+1)*size]; emb=extract_embeddings(model,frame,batch,std,batch_size=8); out=ART/'embeddings'/f'embedding_{n:03d}.npz'; ids=embedding_ids(batch); atomic_npz(out,ids=np.asarray(ids,dtype=f'<U{max(map(len,ids))}'),embeddings=emb); atomic_json(RESULTS/'batches/embeddings'/f'embedding_{n:03d}.json',{'status':'complete','stage':'embedding','batch_number':n,'config_sha256':h,'source_sha256':source,'first_id':ids[0],'last_id':ids[-1],'outputs':[output_record(out,ROOT,len(batch))]}); es[n]=True; write_progress(f'Stage 6 embedding batch {n+1}/12',extra={'embedding_batches_complete':sum(es),'query_batches_complete':sum(query_status(c,h)),'next_command':'python run_decision_aware_darr_phase3.py build --resume'}); print(f'embedding {n+1}/12',flush=True)
    embeddings=all_embeddings(c,h,frame,memory,queries,new); p1=json.loads((ROOT/c['inputs']['phase1_config']).read_text()); std=training_standardizer(frame,train_start='2012-02-01',train_end_exclusive='2015-01-01'); union={window_id(x):x for x in [*memory,*queries]}; union_list=list(union.values()); patterns=context_pattern_matrix(frame,union_list,std,c['memory']['pattern_hours']); pattern_by=dict(zip(embedding_ids(union_list),patterns)); library=minimal_library(frame,memory); retro=pd.read_parquet(ROOT/c['inputs']['retrospective_predictions']); direct=direct_prediction_lookup(retro,c['models']['direct']); qs=query_status(c,h)
    for n,done in enumerate(qs):
        if done: continue
        batch=queries[n*30:(n+1)*30]; qtab=query_table(frame,batch,direct); pairs,audit=candidate_pairs(frame=frame,memory=memory,queries=batch,embedding_by_id=embeddings,pattern_by_id=pattern_by,checkpoint_sha256=p1['model']['head_checkpoint_sha256'],top_k=64,temperature=.05); inference=build_inference_pair_table(pairs,qtab,library); predictions={};
        for key,item in models.items():
            ck=load_checkpoint(ROOT/item['record']['path']); scored=pairs[['query__id','candidate__id']].copy(); scored['score']=predict(ck['model'],ck['experiment'],inference); s=item['selection']; predictions[key]=retrieval_forecasts(scored,library,qtab,score_column='score',k=s['k'],direct_weight=s['direct_weight'])
        pairs['heuristic']=heuristic_scores(pairs); predictions['causal_heuristic']=retrieval_forecasts(pairs,library,qtab,score_column='heuristic',k=8,direct_weight=0); pairs['embedding_score']=-pairs.pair__embedding_rank; predictions['embedding_top1']=retrieval_forecasts(pairs,library,qtab,score_column='embedding_score',k=1,direct_weight=0)
        values=validate_canonical_hourly(frame).price_eur_mwh.to_numpy(float); origin_by={window_id(x):x for x in batch}; rows=[]
        for method,forecasts in predictions.items():
            for qid,forecast in forecasts.items():
                o=origin_by[qid]; actual=values[o.target_start:o.target_start+24]
                for hour in range(24): rows.append({'query__id':qid,'delivery_date':o.delivery_date.date().isoformat(),'hour':hour,'actual':actual[hour],'prediction':forecast[hour],'model':method,'forecast_origin':o.forecast_origin.isoformat()})
        directory=ART/'query_batches'/f'query_{n:03d}'; pred=directory/'predictions.parquet'; aud=directory/'eligibility.parquet'; atomic_parquet(pred,pd.DataFrame(rows)); atomic_parquet(aud,audit); atomic_json(RESULTS/'batches/queries'/f'query_{n:03d}.json',{'status':'complete','stage':'query','batch_number':n,'config_sha256':h,'source_sha256':source,'first_date':audit.delivery_date.min(),'last_date':audit.delivery_date.max(),'queries':len(batch),'outputs':[output_record(pred,ROOT,len(rows)),output_record(aud,ROOT,len(audit))]}); qs[n]=True; write_progress(f'Stage 6 query batch {n+1}/25',extra={'embedding_batches_complete':sum(es),'query_batches_complete':sum(qs),'next_command':'python run_decision_aware_darr_phase3.py build --resume'}); print(f'query {n+1}/25',flush=True)
    write_progress('Stage 6 historical forecasts complete',extra={'embedding_batches_complete':sum(es),'query_batches_complete':sum(qs),'next_command':'python run_decision_aware_darr_phase3.py evaluate --resume'})

def run_status():
    c,h,_,_=frozen_inputs(); _,_,_,new=origins(); es=embed_status(c,h,new); qs=query_status(c,h); final=False
    if (RESULTS/'manifest.json').exists():
        m=json.loads((RESULTS/'manifest.json').read_text()); final=m.get('status')=='complete' and all(verify_output(x,ROOT)[0] for x in m.get('outputs',[]))
    print(json.dumps({'status':'complete' if final and all(es) and all(qs) else 'partial','embedding_batches':f'{sum(es)}/{len(es)}','query_batches':f'{sum(qs)}/25','final_manifest_verified':final,'next_command':'none' if final else 'python run_decision_aware_darr_phase3.py build --resume'},indent=2))

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True); s.add_parser('plan'); b=s.add_parser('build'); b.add_argument('--resume',action='store_true'); s.add_parser('status'); s.add_parser('evaluate'); s.add_parser('audit'); s.add_parser('report'); a=p.parse_args()
    if a.cmd=='plan':run_plan()
    elif a.cmd=='build':run_build(a.resume)
    elif a.cmd=='status':run_status()
    else: raise SystemExit(f'{a.cmd} implemented after forecast batches')
if __name__=='__main__':main()
