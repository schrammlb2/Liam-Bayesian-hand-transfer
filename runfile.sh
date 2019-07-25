# python3 regression_pt.py _ real_A .1 
# python3 regression_pt.py _ real_B .1 

# python3 regression_pt.py _ transferB2A .1 _ retrain
# python3 regression_pt.py _ transferB2A .1 _ constrain_retrain
# python3 regression_pt.py _ transferB2A .1 _ linear_transform
# python3 regression_pt.py _ transferB2A .1 _ single_transform

# python3 regression_pt.py _ transferB2A .1 _ constrained_restart 0.0
# python3 regression_pt.py _ transferB2A .1 _ constrained_restart .0001
# python3 regression_pt.py _ transferB2A .1 _ constrained_restart .0003
# python3 regression_pt.py _ transferB2A .1 _ constrained_restart .001
# python3 regression_pt.py _ transferB2A .1 _ constrained_restart .003
# python3 regression_pt.py _ transferB2A .1 _ constrained_restart .01
# # python3 regression_pt.py _ transferB2A .1 _ constrained_restart .03
# python3 regression_pt.py _ transferB2A .1 _ constrained_restart .1


export method='nonlinear_transform'
python3 regression_pt.py _ transferB2A .99 _ $method
python3 regression_pt.py _ transferB2A .98 _ $method
python3 regression_pt.py _ transferB2A .97 _ $method
python3 regression_pt.py _ transferB2A .96 _ $method
python3 regression_pt.py _ transferB2A .95 _ $method
python3 regression_pt.py _ transferB2A .94 _ $method
python3 regression_pt.py _ transferB2A .93 _ $method
python3 regression_pt.py _ transferB2A .92 _ $method
python3 regression_pt.py _ transferB2A .91 _ $method
# python3 regression_pt.py _ transferB2A .9 _ $method
# python3 regression_pt.py _ transferB2A .8 _ $method
# python3 regression_pt.py _ transferB2A .7 _ $method
# python3 regression_pt.py _ transferB2A .6 _ $method
# python3 regression_pt.py _ transferB2A .5 _ $method
# python3 regression_pt.py _ transferB2A .4 _ $method
# python3 regression_pt.py _ transferB2A .3 _ $method
# python3 regression_pt.py _ transferB2A .2 _ $method
# python3 regression_pt.py _ transferB2A .1 _ $method



# python3 regression_pt.py _ real_A .99 
# python3 regression_pt.py _ real_A .98 
# python3 regression_pt.py _ real_A .97 
# python3 regression_pt.py _ real_A .96 
# python3 regression_pt.py _ real_A .95 
# python3 regression_pt.py _ real_A .94 
# python3 regression_pt.py _ real_A .93 
# python3 regression_pt.py _ real_A .92 
# python3 regression_pt.py _ real_A .91 
# python3 regression_pt.py _ real_A .9 
# python3 regression_pt.py _ real_A .8 
# python3 regression_pt.py _ real_A .7 
# python3 regression_pt.py _ real_A .6 
# python3 regression_pt.py _ real_A .5 
# python3 regression_pt.py _ real_A .4 
# python3 regression_pt.py _ real_A .3 
# python3 regression_pt.py _ real_A .2 
# python3 regression_pt.py _ real_A .1 

python3 learning_curve.py
