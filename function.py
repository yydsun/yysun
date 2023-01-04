
def get_sec(a):
    # Implement a function that return the second max number of array a
    max_num = a[0]
    sec_num = a[0]
    for i in range(len(a)):
        if max_num < a[i]:
            sec_num = max_num
            max_num = a[i]
        elif sec_num < a[i] and a[i] < max_num:
            sec_num = a[i]

    return sec_num

def main():
    a = [1,5,6,6,8,4,1,2,6,4,1,55,6,6,5,4,4,7,8,9,9,1,323,5,654,68,651,65,4,54,54,615,65,6]
    print(get_sec(a))

if __name__ == "__main__":
    main()