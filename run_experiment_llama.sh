python main_vs_baseline.py --user --llm llama2-7b --rule_model cfr > ~/output_cfr.log 2>&1
python main_vs_baseline.py --user --llm llama2-7b --rule_model nfsp > ~/output_nfsp.log 2>&1
python main_vs_baseline.py --user --llm llama2-7b --rule_model dqn > ~/output_dqn.log 2>&1
python main_vs_baseline.py --user --llm llama2-7b --rule_model dmc > ~/output_dmc.log 2>&1