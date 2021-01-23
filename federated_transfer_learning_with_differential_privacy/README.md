### Federated Transfer Learning whit Differential Privacy
- 차등 프라이버시(differential privacy)를 적용한 Federated Transfer Learning
  - 외부 공격(e.g. Model Inversion Attack)에 저항하기 위해 차등 프라이버시를 적용하여 모델을 평가하기 위함
  
#### Server side
- websocket_server.py
  - federated_learning_with_transfer_learning의 websocker_server.py과 다른 점
    - 클라이언트의 모델 학습을 위한 정보 매개변수인 config에서 전달하고자하는 정보가 증가함 (for differential privacy)
  - ε(epsilon) 값을 다르게하여 모델의 정확도를 비교하기 위해 config 내용 수정 필요
- 코드 실행 방법
```
$ python3 websocket_server.py
```

#### Client side
- websocker_client_dp.py
  - 단일 클러스터로 구성되어 있음
  - client_model_dp_opti.py파일을 통해 모델 학습 진행
    - [참고한 differential privacy 적용 코드](https://github.com/lingyunhao/Deep-Learning-with-Differential-Privacy/blob/master/cifar10.py)
- 코드 실행 방법
```
$ python3 websocket_client_dp.py
```

#### 기타
- differential privacy 참고
  - https://github.com/tensorflow/privacy
  - http://www.cleverhans.io/privacy/2019/03/26/machine-learning-with-differential-privacy-in-tensorflow.html?ref=hackernoon.com
