
Requirements:
1. Install dReal3 (https://github.com/dreal/dreal3)
2. Install python (python version > 3.0) packages: 
pip install -U numpy sympy scipy portion intervaltree pathos matplotlib ply pandas scikit-learn

To run:

cd identifiability

For Ex1 (2D, ODE):
python gp_opt/check_box_test_approx.py -c examples/th_ei/config_th.json > logs/log_th.txt

For Ex2 (2D, HS):
python gp_opt/check_box_test_approx.py -c examples/pc/config_pc.json > logs/log_pc.txt

For Ex3 (3D, HS):
python gp_opt/check_box_test_approx.py -c examples/bb/config_bb.json > logs/log_bb.txt

For Ex4 (4D, ODE):
python gp_opt/check_box_test_approx.py -c examples/ex1/config_ex1.json > logs/log_ex1.txt
