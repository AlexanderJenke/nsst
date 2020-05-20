import pickle
import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        print(pickle.load(f)['rules'])