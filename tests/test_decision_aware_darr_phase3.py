from __future__ import annotations
import json
import numpy as np
import pandas as pd

import run_decision_aware_darr_phase3 as runner
from src.research.decision_aware_darr.phase3 import candidate_pairs
from src.research.kumo.data import validate_canonical_hourly

def test_frozen_checkpoints_and_inputs_verify():
    _,_,models,_=runner.frozen_inputs(); assert set(models)=={'ridge_forecast','ridge_economic','lightgbm_forecast','lightgbm_economic'}

def test_development_reproduction_gate_passed():
    report=json.loads((runner.RESULTS/'development_reproduction.json').read_text()); assert report['status']=='passed'; assert max(x['max_abs_prediction_difference'] for x in report['pipelines'])<1e-8

def test_retrospective_dates_and_canonical_products():
    frame,memory,queries,new=runner.origins(); validate_canonical_hourly(frame); assert len(queries)==731; assert len(memory)==2159; assert len(new)==733; assert queries[0].delivery_date==pd.Timestamp('2016-01-01'); assert queries[-1].delivery_date==pd.Timestamp('2017-12-31'); assert all(x.scored_products==24 for x in queries)

def test_memory_is_strictly_causal_for_boundary_queries():
    frame,memory,queries,_=runner.origins(); index=validate_canonical_hourly(frame).index
    from src.research.decision_aware_darr.causal import eligible_candidate_indices
    for query in (queries[0],queries[366],queries[-1]):
        eligible=eligible_candidate_indices(memory,query,index); assert len(eligible)>0
        assert max(memory[i].target_stop for i in eligible)<=query.target_start-12

def test_frozen_feature_order_matches_checkpoint():
    _,_,models,_=runner.frozen_inputs()
    from src.research.decision_aware_darr.reranker import FEATURE_SETS,load_checkpoint
    for item in models.values():
        ck=load_checkpoint(runner.ROOT/item['record']['path']); assert FEATURE_SETS[ck['experiment'].feature_set]

def test_no_retrospective_fit_or_tuning_declared():
    config,_=runner.cfg(); assert config['scope'].startswith('frozen retrospective'); assert config['spread_amplification_control'].startswith('omitted'); assert config['evaluation']['spike_threshold_eur_mwh']==100.0
