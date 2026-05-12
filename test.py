from maraboupy import Marabou
import numpy as np
import time
from sklearn.datasets import load_iris


def verify_sample_robustness(model_path, x_base, y_base, epsilon):
    """
    특정 샘플에 대해 모든 오답 클래스로의 변이 가능성을 검증합니다.
    """
    other_classes = [i for i in range(3) if i != y_base]
    
    for other_class in other_classes:
        # 매번 신규 네트워크 객체 생성 (제약 조건 독립성 보장)
        network = Marabou.read_onnx(model_path)

        # 1. 입력 제약 설정 (L-infinity ball)
        for i in range(len(x_base)):
            network.setLowerBound(i, x_base[i] - epsilon)
            network.setUpperBound(i, x_base[i] + epsilon)

        # 2. 출력 제약 설정 (반례 조건: 정답 클래스가 오답 클래스보다 낮아지는가?)
        #  y_other >= y_base
        network.addInequality([y_base, other_class], [1, -1], 0)

        # 3. 검증 실행
        exit_code, vals, stats = network.solve(verbose=False)
        
        if exit_code == "sat":
            # 반례를 찾은 경우 즉시 반환
            counter_example = [vals[i] for i in range(len(x_base))]
            return "sat", other_class, counter_example

    # 모든 오답 클래스에 대해 반례가 없으면 강건함 증명
    return "unsat", None, None

def main():
    # 1. 설정 및 데이터 로드
    MODEL_PATH = "resources/onnx/iris/iris_model.onnx"
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. 테스트할 샘플 및 마진(epsilon) 설정
    epsilon = 0.8
    sample_idx = 0  # 첫 번째 샘플 사용
    x_test = X[sample_idx]
    y_test = y[sample_idx]
    

    print(f"--- 검증 시작 ---")
    print(f"샘플 인덱스: {sample_idx}")
    print(f"Base Input: {x_test}")
    print(f"Base Label: {y_test}")
    print(f"Epsilon: {epsilon}")

    # 3. 검증 수행 및 시간 측정
    start_time = time.time()
    result, target_class, adv_input = verify_sample_robustness(MODEL_PATH, x_test, y_test, epsilon)
    end_time = time.time()

    # 4. 결과 출력
    print(f"\n--- 결과 보고 ---")
    if result == "sat":
        print(f"결과: [SAT] 강건성 붕괴 발견!")
        print(f"취약 클래스: {target_class}")
        print(f"적대적 반례(Adversarial Input): {adv_input}")
    else:
        print(f"결과: [UNSAT] 해당 범위 내에서 수학적으로 강건함이 증명됨.")
    
    print(f"소요 시간: {end_time - start_time:.4f} 초")

if __name__ == "__main__":
    main()