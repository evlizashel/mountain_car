import os
import sys
import gym
import h5py
from NNmodel import *
import random
import glob
import numpy as np

cd = os.getcwd()


env = gym.make('MountainCar-v0')

#################################################
######## HYPERparameters
#################################################
saving_interval, n_of_episodes = get_params()
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.001
n_hidden = 16

n_of_steps = 200

steps_counter = 1
starting_epsilon = 1.0
ending_epsilon = 0.01
decay_rate = 0.0001
memory = []
memory_size = 16384
batch_size = 64
gamma = 1

#################################################
######## building the model
#################################################


main_net = neural_network_model(input_size,output_size,learning_rate,n_hidden)
target_net = neural_network_model(input_size,output_size,learning_rate, n_hidden)

#################################################
######## Start
#################################################


for episode in range(1, n_of_episodes + 1):
	episode_score = 0
	current_state = env.reset()
	for steps in range(n_of_steps):

		if len(memory) >= memory_size:
			del memory[0]
#################################################
######## Do we want to try something randomly?
#################################################
		
		exploration_prob = ending_epsilon + (starting_epsilon - ending_epsilon) * np.exp(-decay_rate * steps_counter)

		if exploration_prob > np.random.rand():
			action = env.action_space.sample()
		else:
			action = np.argmax(main_net.predict(np.array([current_state]))[0])

#################################################
######## Perform
#################################################

		next_state,reward,done,info = env.step(action)
		
#################################################
######## Assess
#################################################

		if done:
			if next_state[0] >= 0.5:
				memory.append((current_state,action,next_state,1,done))
			else:
				memory.append((current_state,action,next_state,-1,done))
			target_net.set_weights(main_net.get_weights())
			break
		else:
			memory.append((current_state,action,next_state,reward,done))
		current_state = next_state
		
#################################################
######## Adjust
#################################################

		if len(memory) >= batch_size:
			batch = random.sample(memory,batch_size)
			x_train = []
			y_train = []
			for cur_state,act,next_st,rew,complet in batch:
				y = main_net.predict(np.array([cur_state]))[0]
				if complet:
					y[act] = rew
				else:
					y_target = target_net.predict(np.array([next_st]))[0]
					y_profit = rew + gamma * np.max(y_target)
					y[act] = y_profit
				x_train.append(cur_state)
				y_train.append(y)
			main_net.fit(np.array(x_train),np.array(y_train),epochs=1,verbose=0)
		episode_score += reward
		steps_counter += 1

#################################################
######## Output
#################################################

	if episode % saving_interval == 0 and episode != 0 :
		main_net.save_weights("mountain_car_"+str(episode)+".h5")
	if episode % saving_interval == 0 or episode == 1:
		if episode_score > -199:
			winning = ' Pobeda!'
		else:
			winning = ' Porazhenie:('
		print("Episode ", episode, " traininig finished. Episode earned ", episode_score , " points.", winning ) 

print("Training completed")


env.close()

