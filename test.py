from maraboupy import Marabou
import numpy as np
import time
from sklearn.datasets import load_iris

network = Marabou.read_onnx("resources/onnx/iris/iris_model.onnx")

iris = load_iris()
X, y = iris.data, iris.target

x_base = X[0]  # Base input (first sample)
epsilon = 0.000005 # perturbation radius

for i in range(4):
    network.setLowerBound(i, x_base[i] - epsilon)
    network.setUpperBound(i, x_base[i] + epsilon)


network.addInequality([1, 0], [1, -1], 0) 

start_time = time.time()
exit_code, vals, stats = network.solve()
end_time = time.time()

print(f"Verification Result: {exit_code}")
print(f"Time taken: {end_time - start_time:.4f} seconds")

if exit_code == "sat":
    print("Counter-example found (Adversarial Input):")
    print([vals[i] for i in range(4)])