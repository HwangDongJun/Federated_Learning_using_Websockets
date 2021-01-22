# Federated Learning using Websockets
- python websockets, asyncio 모듈을 사용한 비동기 방식 Federated Learning 구현 (서버와 클라이언트의 통신상에서 발생할 수 있는 예외처리 미구현)
  - Federated Learning을 사용한 사용자 활동 인식 데이터 보호 (개인 정보 보호)
  - 클라이언트마다 다른 데이터 특징과 적은 데이터의 문제점을 보완하기 위한 Transfer Learning 적용

## Getting Started (Ubuntu 환경)
- python websockets, asyncio 모듈 설치 필수
```
> websockets 설치
$ pip3 install websockets
> asyncio 설치
$ pip3 install asyncio
```
- tensorflow (≥ 2.4.0), tensorflow hub (≥ 0.10.0)
```
> tensorflow 설치
$ pip3 install tensorflow
> tensorflow-hub 설치
$ pip3 install tensorflow-hub
```

- Dictionary structure
  - federated_learning_with_transfer_learning
  - federated_learning_without_transfer_learning
  - federated_transfer_learning_with_differential_privacy

## 주의사항
- Federated Learning을 위한 통신 구현 중 클라이언트에서 발생할 수 있는 예외 상황은 제외한 구현
  - 정상 동작만을 가정한 구현을 의미
- 현재 구현 중임을 알림
  - 2020.07.05 ~
    - with transfer learning, without transfer learning, differntial privacy 코드파일 추가
  - 2021.01.22 ~
    - with transfer learning 코드 파일 및 설명 하나의 폴더로 분리

