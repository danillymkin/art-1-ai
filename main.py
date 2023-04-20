from NetworkART import NetworkART
import numpy as np


def run_net(vectors, P, L):
    net = NetworkART(vectors[0], P, L)

    for vector in vectors[1:]:
        net.recognition(vector)


def main():
    vectors_test = np.array([[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                             [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                             [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                             [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]])
    run_net(vectors_test, 0.8, 2)


if __name__ == '__main__':
    main()
