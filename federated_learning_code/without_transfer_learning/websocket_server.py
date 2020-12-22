import os
import asyncio
import binascii
import websockets
import pickle
import json
import numpy as np
from base64 import b64encode

import time
from threading import Thread


class PythonObjectEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
			return super().default(obj)
		return {'_python_object': b64encode(pickle.dumps(obj)).decode('utf-8')}

class Server:
	def __init__(self, loop=None):
		self.time = time.time()
		#self.connection_queue = asyncio.Queue()
		self.configuration_queue = asyncio.Queue()
		self.train_aggregation_queue = asyncio.Queue()
		self.first_conn_count = 0; self.check_client = 1
		self.timer_check = True

		if loop is None:
			loop = asyncio.new_event_loop()
		self.loop = loop
		#self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
		self.USERS = set()
		self.len_USERS = 0

		self.client_count = 0;
		self.aggregation_list = list(); self.averaged_weight = list()

	def timer(self, checker):
		time.sleep(10) # 임시로 10초동안 aggregation을 함
		self.timer_check = False
		checker.insert(0, self.timer_check)

	async def _check_count(self):
		self.check_client = self.first_conn_count - 1
		while True:
			if self.first_conn_count == self.check_client:
				await asyncio.sleep(0.1)
				break

	async def _consumer_handler(self, websocket: websockets.WebSocketCommonProtocol):
		# the connection object to receive messages from and add them ito the queue.
		try:
			while True:
				#if self_first_conn_count == self.check_client:
					#await asyncio.sleep(N)
	
				# conn_list :연결된 client의 list생성
				msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
				#msg = await websocket.recv()
				# register websocket
				await websocket.send(msg)
				self.USERS.add(websocket)
				#if msg == 'EF' or msg == 'NEF':
					#for i in range(self.first_conn_count):
						#print("finished client selection! -> wait recv weight")
						#msg = await websocket.recv()
						#await self.train_aggregation_queue.put(msg)
						#await self._check_count()
					#break
				self.first_conn_count += 1
				print("connected websocket!")
				#await websocket.send(msg)
				#await self.configuration_queue.put(msg)
		except: # timeout에 관련된 예외처리
			print("finished! connected client!", str(self.first_conn_count))

	#async def _producer_handler(self, websocket: websockets.WebSocketCommonProtocol):
		#while True:
			# get a message from the queue
			#message = await self.connection_queue.get()
			# send the message
			#await websocket.send(message)
			#await self.configuration_queue.put(message)
	
	async def _configuration_handler(self, websocket: websockets.WebSocketCommonProtocol):
		for i in range(self.first_conn_count):
			try:
				user = self.USERS.pop()
				print("User websock id")
				print(user)
				config = '224,32,0.001,10,adam,https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
				#config = '224,32,0.001,10,adam,https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1'
				#hex_config = binascii.hexlify(config.encode())
				# 기존에 학습한 파일이 있는지 확인
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
			with open('./model_weights/weights.pickle', 'rb') as f:
				previous_weight = pickle.load(f)
			return previous_weight
		except (OSError, IOError) as e:
			return list()

	async def _aggregation_parameter(self, websocket: websockets.WebSocketCommonProtocol):
		# timer발동
		#check_list = [True]
		#th1 = Thread(target=timer, args=(check_list,))
		#th1.start()
		
		weight_crawler_list = list()
		# 이 함수가 클라이언트들이 학습을 끝낸 다음 파라미터를 전달했을 경우 합치는 함수
		for user in self.USERS_for_aggregation:
			weight_parameter = await user.recv()
			weight_decoding = pickle.loads(weight_parameter)
			print(weight_decoding)
			weight_crawler_list.append(weight_decoding)

		print(len(weight_crawler_list))

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
			#if self.client_count == 1:
				#self.averaged_weight = parameter_decoding
			#else:
				#for i, wcl in enumerate(parameter_decoding):
					#self.averaged_weight[i] = self.averaged_weight[i] + wcl

			if self.client_count == 3: # 3은 인위적으로 넣은 값 수정해야함
				print(self.client_count)
				break
				
		for wcl in self.aggregation_list:
			if len(self.averaged_weight) == 0:
				self.averaged_weight = wcl
			else:
				for i, wc in enumerate(wcl):
					self.averaged_weight[i] = self.averaged_weight[i] + wc
		
		# devide n client
		client_len = 3 # 3은 인위적으로 넣은 값 수정해야함
		for i, aw in enumerate(self.averaged_weight):
			self.averaged_weight[i] = aw / client_len

		print("complete aggregation parameter and save to file!")
		with open('./model_weights/weights.pickle', 'wb') as fw:
			pickle.dump(self.averaged_weight, fw)

		print("finished total process... time is " + str(time.time() - self.time))
		#os.exit()
				
	async def _handler(self, websocket: websockets.WebSocketCommonProtocol, *unused_args):
		try:
			asyncio.set_event_loop(self.loop)
	
			await self._consumer_handler(websocket)
			print("===== Finish client selection =====")
			await asyncio.sleep(5.0)
			#await self._configuration_handler(websocket)
			await asyncio.wait_for(self._configuration_handler(websocket), timeout=10.0)
			await self._configuration_handler(websocket)
			print("===== Finish transfer configuration information =====")
			#await self._aggregation_parameter(websocket)
	
			precv_task = asyncio.ensure_future(self._parameter_recv(websocket))
			aggregation_task = asyncio.ensure_future(self._aggregation(websocket))
		
			done, pending = await asyncio.wait(
				[precv_task, aggregation_task], return_when=asyncio.FIRST_EXCEPTION
			)
	
			for task in pending:
				task.cancel()
		except asyncio.TimeoutError:
			print("===== Finish transfer configuration information =====")
			precv_task = asyncio.ensure_future(self._parameter_recv(websocket))
			aggregation_task = asyncio.ensure_future(self._aggregation(websocket))

			done, pending = await asyncio.wait(
				[precv_task, aggregation_task], return_when=asyncio.FIRST_EXCEPTION
			)

			for task in pending:
				task.cancel()

	def start(self):
		return websockets.serve(self._handler, "localhost", 8000, ping_interval=None, ping_timeout=None, max_size=2**29)

# waiting
if __name__ == '__main__':
	ws = Server()
	asyncio.get_event_loop().run_until_complete(ws.start())
	asyncio.get_event_loop().run_forever()
