# 신뢰할 수 있는 인공지능 과제 3
이 저장소는 Marabou를 활용한 신경망 검증 과제 수행 기록입니다.
## How to Run
### Install
저장소의 코드를 Clone할 수 있다.
```bash
git clone https://github.com/paul6598/ReliableAI_3.git
```

pip을 통해 Marabou를 설치할 수 있다.
```bash
pip install maraboupy
```

또한 requirements.txt에서 추가적인 의존성을 설치할 수 있다.
```bash
pip install -r requirements.txt
```

### Run
실행은 저장소 코드의 test.py을 통해 이루어진다.
터미널에 해당 line을 입력할 수 있다.

```bash
python test.py
```

#### hyperparameter
test.py 파일의 line 44, 45의 epsilon, sample_idx를 조절하여 노이즈의 마진, 테스트할 샘플의 index를 지정할  수  있다.