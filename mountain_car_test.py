import os
import sys
import gym
import h5py
from NNmodel import *
import random
import glob
import numpy as np


#################################################
######## Get the model
#################################################
saving_interval, n_of_episodes = get_params()

def input_function(message):
	index = input(message)
	if len(index) == 0:
		index = input_function(" Wrong input, try again. \n Choose the model: input the index of the preferred model to test: ")
	elif sum([1 for x in index if x not in ["0","1","2","3","4","5","6","7","8","9"]]) > 0:
		index = input_function(" Index must be numeric, try again. \n Choose the model: input the index of the preferred model to test: ")
	elif int(index) % saving_interval != 0 or int(index) > n_of_episodes:
		index = input_function(" Index must be devisible by " + str(saving_interval) + " and cannot be above " + str(n_of_episodes) + " try again. \n Choose the model: input the index of the preferred model to test: ")
	return index

index = input_function("Choose the model: input the index of the preferred model to test: ")
cd = os.getcwd()
env = gym.make('MountainCar-v0')

#################################################
######## HYPERparameters
#################################################

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.001
n_hidden = 16
n_of_episodes_test = 20
n_of_steps = 200


#################################################
######## Build the net
#################################################

main_net = neural_network_model(input_size,output_size,learning_rate, n_hidden)

#################################################
######## Load the net
#################################################

main_net.load_weights(cd+"/mountain_car_" + index + ".h5")


#################################################
######## Testing
#################################################

for episode in range(n_of_episodes_test):
	current_state = env.reset()
	episode_score = 0 
	for steps in range(n_of_steps):
#################################################
######## Build the image
#################################################
		env.render()
		action = np.argmax(main_net.predict(np.array([current_state]))[0])
		next_st,rew,complet,info = env.step(action)
		if complet:
			break
		current_state = next_st
		episode_score += rew

#################################################
######## Output
#################################################

	print("Episode ", episode+1, " tested. Episode earned ", episode_score , " points.") 

print("Testing completed")

env.close()
