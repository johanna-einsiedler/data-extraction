#!/bin/bash
cd /Users/einsie0004/Documents/research/23_paper_extraction/pipeline_june_26
pipeline_env/bin/python analysis/hpo_analysis/rescore_aug_batch_phi3.py analysis/cache/mode_by_type/indiv_all 335 347 351 391 392 393 394 395 412 414 >> analysis/cache/mode_by_type/indiv_all/chain.log 2>&1
echo INDIV_DONE >> analysis/cache/mode_by_type/indiv_all/chain.log
