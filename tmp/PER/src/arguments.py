import argparse


def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    parser.add_argument('--lr', metavar='L', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--y', metavar='Y', type=float, default=0.95,
                        help='discount factor')
    parser.add_argument('--e', metavar='E', type=float, default=0.998,
                        help='e-greedy factor')
    parser.add_argument('--r', metavar='R', type=int, default=2000,
                        help='total episodes (rounds)')
    parser.add_argument('--s', metavar='S', type=int, default=100,
                        help='total steps per episode')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    print(args)
