### Federated Learning without Transfer Learning
- Transfer Learning을 적용하지 않은 상태에서 일반적인 학습을 진행
  - 적용 모델 -> MobileNet V2 / EfficientNet B0
 
 #### Server side
 - 서버의 경우 [federated_learning_with_transfer_leanring의 federated_server.py](https://github.com/HwangDongJun/Federated_Learning_using_Websockets/blob/master/federated_learning_with_transfer_learning/federated_server.py)과 코드가 동일하므로 추가하지 않음
 
 #### Client side
 - federated_client.py
  - 단일 클러스터로 구성되어 있음
  - ntf_client_fit_model.py파일을 통해 모델 학습 진행
    - 적용 모델을 변경하고자할 경우 해당 파일에서 코드 수정 필요
- 코드 실행 방법
```
$ python3 federated_client.py
```
