
SMT based parameter identifiable combination detection

Requirements:
1. dReal3 (https://github.com/dreal/dreal3)
2. python (python version > 3.6)
3. python packages: numpy sympy scipy portion intervaltree pathos matplotlib ply pandas scikit-learn scikit-learn-extra
    -- installation through conda (https://developers.google.com/earth-engine/guides/python_install-conda):	
		conda install -c conda-forge numpy sympy scipy portion intervaltree pathos matplotlib ply pandas scikit-learn scikit-learn-extra

To run:

$> cd identifiability

For Example 1 (2D, ODE):

$> python gp_opt/check_box_test_approx.py -c examples/th_ei/config_th.json > logs/log_th.txt

For Example 2 (2D, HS):

$> python gp_opt/check_box_test_approx.py -c examples/pc/config_pc.json > logs/log_pc.txt

For Example 3 (3D, HS):

$> python gp_opt/check_box_test_approx.py -c examples/bb/config_bb.json > logs/log_bb.txt

For Example 4 (4D, ODE):

$> python gp_opt/check_box_test_approx.py -c examples/ex1/config_ex1.json > logs/log_ex1.txt
