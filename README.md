GitHub link: https://github.com/CMBeckers/Assignment_end_of_april.git

Requirements:
Install numpy
Install matplotlib
Install random
Install argparse
Install math


How to run:
Open your terminal and using the cd command, navigate to the folder where you stored the assignments' file. 
Ensure that you store the file as a .py file

To run any of the flags to be stated, you must enter one of the following into the command line:
	python assignment.py <flag>
	python3 assignment.py <flag>
	

For Task 1.) you can enter the following flags into the command line to run this model:
	-ising_model #This should run the ising model with default parameters
	-ising_model -external -0.1 #This should run the ising model with default temperature and an external influence of -0.1
	-ising_model -alpha 10 #This should run the ising model with no external influence but with a temperature of 10 degrees
	-test_ising #This should run the test functions associated with the model. 


For Task 2.) you can enter the following flags into the command line to run this model:
	-defuant #This should run the defuant model with default parameters
	-defuant -beta 0.1 #This should run the defuant model with default threshold and a beta of 0.1.
	-defuant -threshold 0.3 #This should run the defuant model with a threshold of 0.3.
	-test_defuant #This should run the test functions that we have written.


For Task 3.) you can enter the following flags into the command line to run this model:
	-network 10 #This should create and plot a random network of size 10, it will also return the mean Degree, average path length and clustering co-efficient
	-test_network #This should run the test functions provided


For Task 4.) you can enter the following flags into the command line to run this model:
	-ring_network 10 # This should create a ring network with a range of 1 and a size of 10
	-small_world 10 #This should create a small-worlds network with default parameters
	-small_world 10 -re_wire 0.1 #This should create a small worlds network with a re-wiring probability of 0.1


For Task 5.) you can enter the following flags into the command line to run this model:
	-defuant -use_network 10 #This should solve the defuant model on a small world network of size 10.
 	-defuant -use_network #The default value is set to 100, which gives a better clearer and graph.



