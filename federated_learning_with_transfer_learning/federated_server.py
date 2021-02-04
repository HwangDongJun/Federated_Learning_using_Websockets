import asyncio
import binascii
import websockets
import pickle
import json
import numpy as np
import time


class PythonObjectEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
			return super().default(obj)
		return {'_python_object': b64encode(pickle.dumps(obj)).decode('utf-8')}

class Server:
	def __init__(self, loop=None):
		self.configuration_queue = asyncio.Queue()
		self.train_aggregation_queue = asyncio.Queue()
		self.first_conn_count = 0

		if loop is None:
			loop = asyncio.new_event_loop()
		self.loop = loop
		self.USERS = set()

		self.client_count = 0;
		self.aggregation_list = list(); self.averaged_weight = list()

	async def _consumer_handler(self, websocket: websockets.WebSocketCommonProtocol):
		# the connection object to receive messages from and add them ito the queue.
		try:
			while True:
				# conn_list : Create a list of connected clients
				msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
				# register websocket
				await websocket.send(msg)
				self.USERS.add(websocket)
				self.first_conn_count += 1
				print("connected websocket!")
		except asyncio.TimeoutError:
			print("finished! connected client!", str(self.first_conn_count))

	async def _configuration_handler(self, websocket: websockets.WebSocketCommonProtocol):
		for i in range(self.first_conn_count):
			try:
				user = self.USERS.pop()
				print("User websocket id")
				print(user)

				# config : Information needed to train the model to send to the clinets.
				# image_shape, batch_size, learning_rate, epochs, optimizer_method, tensorflow_hub_model_link
				config = '96,32,0.002,1,adam,https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/4'
				# Check that you have previously learned files.
				pre_weight = await self._weight_file_check()
	
				if pre_weight == list():
					wrap_json_configuration = json.dumps({
								'config': config, 
								'previous_weight': pre_weight})
				else:
					wrap_json_configuration = json.dumps({
								'config': config,
								'previous_weight': pre_weight}, cls=PythonObjectEncoder)
	
				print("===== Send to client configuration information! =====")
				await user.send(wrap_json_configuration)
			except KeyError:
				break
	
	async def _weight_file_check(self):
		try:
			with open('PREVIOUS WEIGHT FILE PATH(.pickle)', 'rb') as f:
				previous_weight = pickle.load(f)
			return previous_weight
		except (OSError, IOError) as e:
			return list()

	async def _parameter_recv(self, websocket: websockets.WebSocketCommonProtocol):
		while True:
			print("===== wait after model training parameter =====")
			parameter_msg = await websocket.recv()
			print("===== recv parameter -> put queue =====")
			await self.train_aggregation_queue.put(parameter_msg)

	async def _aggregation(self, websocket: websockets.WebSocketCommonProtocol):
		is_parameter_check = await self._weight_file_check()
		if is_parameter_check != list():
			averaged_weight = is_parameter_check

		while True:
			par_msg = await self.train_aggregation_queue.get()
			self.client_count += 1
			parameter_decoding = pickle.loads(par_msg)
		
			self.aggregation_list.append(parameter_decoding)

			# It changes flexibly depending on the communicating client,
			# but it is possible to experiment by fixing the vlaue.
			# e.g. len(self.USERS) -> 3 (fixed client count)
			if self.client_count == len(self.USERS):
				break
				
		for wcl in self.aggregation_list:
			if len(self.averaged_weight) == 0:
				self.averaged_weight = wcl
			else:
				for i, wc in enumerate(wcl):
					self.averaged_weight[i] = self.averaged_weight[i] + wc
		
		# devide n client
		# As described above, the value can be fixed.
		client_len = len(self.USERS)
		for i, aw in enumerate(self.averaged_weight):
			self.averaged_weight[i] = aw / client_len

		print("complete aggregation parameter and save to file!")
		with open('PARAMETER SAVE TO FILE PATH(.pickle)', 'wb') as fw:
			pickle.dump(self.averaged_weight, fw)
				
	async def _handler(self, websocket: websockets.WebSocketCommonProtocol, *unused_args):
		asyncio.set_event_loop(self.loop)

		await self._consumer_handler(websocket)
		print("===== Finish client selection =====")
		await asyncio.sleep(5.0)
		await asyncio.wait_for(self._configuration_handler(websocket), timeout=10.0)
		await self._configuration_handler(websocket)
		print("===== Finish transfer configuration information =====")

		precv_task = asyncio.ensure_future(self._parameter_recv(websocket))
		aggregation_task = asyncio.ensure_future(self._aggregation(websocket))

		done, pending = await asyncio.wait(
			[precv_task, aggregation_task], return_when=asyncio.FIRST_EXCEPTION
		)

		for task in pending:
			task.cancel()

	def start(self):
		# Since the size of the weight file is different for each model, modify the size
		# of max_size when an error occurs.
		# In the example, designate "localhost", and input IP address when using an
		# external server.
		return websockets.serve(self._handler, "localhost", 8000, ping_interval=None, ping_timeout=None, max_size=2**29)

if __name__ == '__main__':
	ws = Server()
	asyncio.get_event_loop().run_until_complete(ws.start())
	asyncio.get_event_loop().run_forever()
