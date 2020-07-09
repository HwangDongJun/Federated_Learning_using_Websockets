import asyncio
import binascii
import websockets
import pickle
import json
import numpy as np

from realtime_object_detection_yolo import object_detection
from client_fit_model import transfer_learning_fit


# training model
async def train_model(state, config):
	print("training client model!")
	train_process = transfer_learning_fit(config, state)
	return train_process.manage_train()

# get configuration from server
async def check_configuration(config_msg):
	#unhex_config = binascii.unhexlify(config_msg)
	check_info = config_msg.split(',')
	model_info = {
			'image_shape' : int(check_info[0]),
			'batch_size' : int(check_info[1]),
			'learning_rate' : float(check_info[2]),
			'epochs' : int(check_info[3]),
			'optimizer': check_info[4],
			'model' : check_info[5]
	}
	return model_info

# send num 1 -> check connect to server
async def check_connect():
	get_weights = list()
	async with websockets.connect("ws://localhost:8000", max_size=2**25) as websocket:
		await asyncio.sleep(2)
		await websocket.send("1")
		right_connection = await websocket.recv()
		print(right_connection)
		if right_connection == "1":
			config_msg = await websocket.recv() # get number 1 -> next recv configuration
			decode_config_msg = json.loads(config_msg)

			model_configuration = await check_configuration(decode_config_msg['config'])

			if decode_config_msg['previous_weight'] != list():
				previous_weight_info = np.asarray(decode_config_msg['previous_weight'])
			else:
				previous_weight_info = list()

			get_weights = await train_model(previous_weight_info, model_configuration)
			
			print(get_weights[-1])

			weight_encoding = pickle.dumps(get_weights)
			await websocket.send(weight_encoding)	

# connect to server
if __name__ == '__main__':
	asyncio.get_event_loop().run_until_complete(check_connect())
	asyncio.get_event_loop().run_forever()
