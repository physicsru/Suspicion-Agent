cuda_visible_devices=0 python main_vs_baseline.py --user --llm agentlm-7b --rule_model cfr --verbose_print > ~/output_cfr_agentlm.log 2>&1
cuda_visible_devices=1 python main_vs_baseline.py --user --llm llama2-7b --rule_model nfsp --verbose_print > ~/output_nfsp_v2.log 2>&1
cuda_visible_devices=2 python main_vs_baseline.py --user --llm llama2-7b --rule_model dqn --verbose_print > ~/output_dqn_v2.log 2>&1
python main_vs_baseline.py --user --llm llama2-7b --rule_model dmc --verbose_print > ~/output_dmc.log 2>&1

cuda_visible_devices=0 python main_vs_baseline.py --user --llm agentlm-13b --rule_model cfr --verbose_print > ~/output_cfr_agentlm_13b.log 2>&1


cuda_visible_devices=0 python main_vs_baseline.py --user --llm chatglm-6b --rule_model cfr --verbose_print > ~/output_cfr_chatglm_6b.log 2>&1


cuda_visible_devices=0 python main_vs_baseline.py --user --llm zerooneai-6b --rule_model cfr --verbose_print > ~/output_cfr_zerooneai_6b.log 2>&1


cuda_visible_devices=0 python main_vs_baseline.py --user --llm zerooneai-6b --rule_model cfr --verbose_print > ~/output_cfr_zerooneai_6b.log 2>&1