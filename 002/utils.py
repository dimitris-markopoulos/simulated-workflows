import random
import math
import numpy as np

def generate_data():
    
    random.seed(0)

    ## Data set 1
    X1 = []
    for i in range(1000):
        theta = random.uniform(0, 2 * math.pi)
        radius = random.gauss(0, 0.2) + random.choice([1, 3])
        X1.append([radius * math.cos(theta), radius * math.sin(theta)])
    X1 = np.array(X1)

    ## Data set 2
    X2 = []
    for i in range(1000):
        theta = random.uniform(0, 2 * math.pi)
        radius = random.gauss(0, 0.1) + 2
        if theta < math.pi:
            X2.append([radius * math.cos(theta) - 1, radius * math.sin(theta)])
        else:
            X2.append([radius * math.cos(theta) + 1, radius * math.sin(theta)])
    X2 = np.array(X2)

    ## Data set 3
    X3 = []
    for i in range(1000):
        radius = random.gauss(0, 1)
        theta = random.uniform(0, 2 * math.pi)
        center = random.choice([[0, 1], [3, 3], [1, -3]])
        X3.append([radius * math.cos(theta) + center[0],
                radius * math.sin(theta) + center[1]])
    X3 = np.array(X3)
    
    return [X1,X2,X3]
