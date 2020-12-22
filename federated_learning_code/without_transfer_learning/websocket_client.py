import asyncio
import binascii
import websockets
import pickle
import json
import numpy as np
import argparse
from base64 import b64decode

from realtime_object_detection_yolo import object_detection
from ntf_client_fit_model import transfer_learning_fit


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

def as_python_object(dct):
	if '_python_object' in dct:
		return pickle.loads(b64decode(dct['_python_object'].encode('utf-8')))
	return dct

# send num 1 -> check connect to server
async def check_connect():
	get_weights = list()
	#try:
	async with websockets.connect("ws://localhost:8000", max_size=2**29) as websocket:
		await asyncio.sleep(1)
		await websocket.send("1")
		right_connection = await websocket.recv()
		if right_connection == "1":
			config_msg = await websocket.recv() # get number 1 -> next recv configuration
			print("received config_msg from server!")
			decode_config_msg = json.loads(config_msg, object_hook=as_python_object)

			model_configuration = await check_configuration(decode_config_msg['config'])
			
			if decode_config_msg['previous_weight'] != list():
				previous_weight_info = np.asarray(decode_config_msg['previous_weight'])
			else:
				previous_weight_info = list()
			
			get_weights = await train_model(previous_weight_info, model_configuration)
			
			weight_encoding = pickle.dumps(get_weights)
			await websocket.send(weight_encoding)
			await websocket.recv()
	#except websockets.exceptions.ConnectionClosed:
		#print("exception ConnectionClosed!!")
		#async with websockets.connect("ws://localhost:8000") as websocket:
			#weight_encoding = pickle.dumps(get_weights)
			#await asyncio.sleep(0.1)
			#await websocket.send(weight_encoding)

# connect to server
if __name__ == '__main__':
	asyncio.get_event_loop().run_until_complete(check_connect())
	#asyncio.get_event_loop().run_forever()
