# python3 regression_general.py pos sim_B 1 transfer_learning_rates False .1
# python3 regression_general load sim_B 1 transfer_learning_rates True .1
# python3 regression_general.py pos sim 1 transfer_learning_rates True .1
# python3 regression_general load sim 1 transfer_learning_rates True .1
# python3 transfer_regression.py pos sim 1 transfer_learning_rates True .1
# python3 transfer_regression load 1 transfer_learning_rates True .1
# python3 transfer_regression.py pos sim 1 transfer_learning_rates True .9
# python3 transfer_regression load 1 transfer_learning_rates True .9


python3 regression_general.py load real_A 
python3 regression_general.py pos real_A
# python3 transfer_regression.py load real
# python3 transfer_regression.py pos real 
python3 prediction_transfer.py real

