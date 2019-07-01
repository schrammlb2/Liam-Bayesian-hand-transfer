# python3 prediction_pt.py _ real_A 
# python3 prediction_pt.py _ real_B 

# python3 prediction_pt.py _ transferA2B _ retrain
# python3 prediction_pt.py _ transferA2B _ constrain_retrain
# # python3 prediction_pt.py _ transferA2B _ linear_transform
# python3 prediction_pt.py _ transferA2B _ single_transform
# # python3 prediction_pt.py _ transferA2B _ constrain_restart

python3 prediction_pt.py _ transferA2B _ constrained_restart 0.0
python3 prediction_pt.py _ transferA2B _ constrained_restart .0001
python3 prediction_pt.py _ transferA2B _ constrained_restart .0003
python3 prediction_pt.py _ transferA2B _ constrained_restart .001
python3 prediction_pt.py _ transferA2B _ constrained_restart .003
python3 prediction_pt.py _ transferA2B _ constrained_restart .01
python3 prediction_pt.py _ transferA2B _ constrained_restart .1

# echo 'after this, should make a runfile for the prediction and run that to collect the data'
# echo 'Then, run the experiment again with no weight decay or gradient norm clipping and see how the model transfers by comparison'
# echo 'You can compare transfer effectiveness based on gradient norm and '
