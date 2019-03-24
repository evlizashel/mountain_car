
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense


n_of_episodes = 500
saving_interval = 20
def get_params():
	return saving_interval, n_of_episodes

def neural_network_model(input_size,output_size,learning_rate, n_hidden):

	model = Sequential()
	model.add(Dense(n_hidden,input_dim=input_size,activation='relu'))
	model.add(Dense(n_hidden,activation='relu'))
	model.add(Dense(output_size,activation='linear'))

	model.compile(loss='mse',optimizer=Adam(lr=learning_rate))

	return model
