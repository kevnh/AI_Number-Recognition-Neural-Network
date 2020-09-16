
import mnist_loader
import neural_net
import argparse
import numpy as np

from imageio import imread


layer_nodes = [784, 30, 10]     # Number of nodes per layer

# Main function
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-u', '--user', action='store_true', default=False,
                        help='Input correct value for image.png')
    parser.add_argument('-a', '--alpha', action='store', type=float,
                        help='Training rate', default=3.0)
    parser.add_argument('-e', '--epochs', action='store', type=int,
                        help='Number of epochs', default=30)
    parser.add_argument('-r', '--random', action='store_true', default=False,
                        help='Randomly initialize weights')
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='Set test mode')
    parser.add_argument('-s', '--save', action='store_true', default=False,
                        help='Save weights/biases')
    parser.add_argument('-b', '--batch', action='store', type=int,
                        help='Batch size', default=10)

    args = parser.parse_args()

    net = neural_net.Network(layer_nodes, args.random)

    if args.user:   # Uses user image and runs it through the network
        im = imread("image.png")
        arr = (np.reshape(im, (784,1)) / -255) + 1
        results = net.user_eval(arr)
        print(results)
    else:
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        if args.test:
            test_size = len(test_data)
            results = net.evaluate(test_data)

            print("Results: %d / %d\n" % (results, test_size))
        else:
            net.SGD(training_data, args.epochs, args.batch, args.alpha, test_data=test_data)

        if args.save:
            net.save_file()


if __name__ == "__main__":
    main()