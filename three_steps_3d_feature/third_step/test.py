import numpy as np
import time

# Simulate list of vectors
lst = [np.random.rand(32).astype(np.float32) for _ in range(98330)]

t1 = time.time()
arr = np.array(lst)
t2 = time.time()

print(f"Conversion time: {t2 - t1:.4f} seconds")