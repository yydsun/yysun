from tempfile import tempdir
import torch
import numpy as np
import pandas as pd

# assume array a[10] = {9,0,4,5,6,3,2,7,8,1}
# implement a function my_sort(a) to sort a[] in asc order

def my_sort(a):
    # Your work should be done here
    for i in range(len(a)):
        min_num = a[i]
        index = i
        j = i + 1
        while j < len(a):
            if min_num > a[j]:
                min_num = a[j]    
                index = j            
            j += 1
        temp = a[i]
        a[i] = a[index]
        a[index] = temp
        
def main():
    a = [9,0,4,5,6,3,2,7,8,1]
    my_sort(a)
    print(a)


if __name__ == "__main__":
    main()