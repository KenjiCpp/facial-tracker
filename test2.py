from cv2 import CAP_PROP_XI_COLUMN_FPN_CORRECTION
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

def func(a):
    return 1, 2

if __name__ == "__main__":
    with Pool(cpu_count() - 1) as p:
        r = p.map(func, range(100))
    for a, b in r:
        print(a, b)