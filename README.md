Anatomical Landmark Detection

Automatic detection of anatomical landmarks is an important step for a wide range of applications in medical image analysis. 
In this project, we formulate the landmark detection problem as a sequential decision process navigating in a facial image towards the target landmark.
We can use this algorithm on real MRI and CT images.
We deploy Deep Q-Network (DQN) based architectures to train agent that can learn to identify the optimal path to the point of interest. 
 
 
Usage: Below command can be used to invoke the program. This command will initiate the training process and save the learning model in the local directory
After training, testing process will load the model from the directory and execute the test part of the program.

python Landmark.py
