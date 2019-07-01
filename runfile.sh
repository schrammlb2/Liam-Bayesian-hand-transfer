# python3 regression_pt_batch2.py pos real_A .1 
# python3 regression_pt_batch2.py pos real_B .1 

# python3 regression_pt_batch2.py pos transferA2B .1 _ retrain
# python3 regression_pt_batch2.py pos transferA2B .1 _ constrain_retrain
# python3 regression_pt_batch2.py pos transferA2B .1 _ linear_transform
# python3 regression_pt_batch2.py pos transferA2B .1 _ single_transform

# python3 regression_pt_batch2.py pos transferA2B .1 _ constrained_restart 0.0
# python3 regression_pt_batch2.py pos transferA2B .1 _ constrained_restart .0001
# python3 regression_pt_batch2.py pos transferA2B .1 _ constrained_restart .0003
# python3 regression_pt_batch2.py pos transferA2B .1 _ constrained_restart .001
# python3 regression_pt_batch2.py pos transferA2B .1 _ constrained_restart .003
# python3 regression_pt_batch2.py pos transferA2B .1 _ constrained_restart .01
# # python3 regression_pt_batch2.py pos transferA2B .1 _ constrained_restart .03
# python3 regression_pt_batch2.py pos transferA2B .1 _ constrained_restart .1




python3 regression_pt_batch2.py pos transferA2B .9 _ constrained_restart 0.0
python3 regression_pt_batch2.py pos transferA2B .9 _ constrained_restart .0001
python3 regression_pt_batch2.py pos transferA2B .9 _ constrained_restart .0003
python3 regression_pt_batch2.py pos transferA2B .9 _ constrained_restart .001
python3 regression_pt_batch2.py pos transferA2B .9 _ constrained_restart .003
python3 regression_pt_batch2.py pos transferA2B .9 _ constrained_restart .01
# python3 regression_pt_batch2.py pos transferA2B .9 _ constrained_restart .03
python3 regression_pt_batch2.py pos transferA2B .9 _ constrained_restart .1

echo 'after this, should make a runfile for the prediction and run that to collect the data'
echo 'Then, run the experiment again with no weight decay or gradient norm clipping and see how the model transfers by comparison'
echo 'You can compare transfer effectiveness based on gradient norm and '
