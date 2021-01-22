### Federated Learning with Transfer Learning
- Federated Transfer Learning을 위한 Server와 Client 및 Learning code 파일

#### Server side
- websocket_server.py
  - 코드 수정 사항
    - 경로 설정 및 학습을 위한 config 변수 수정 필요
- 코드 실행 방법
```
$ python3 websocket_server.py
```

#### Client side
- websocket_client.py
  - 단일 클라이언트로 구성되어 있음
  - client_fit_model.py파일을 통해 모델 학습 진행
- 코드 실행 방법
```
$ python3 websocket_client.py
```
