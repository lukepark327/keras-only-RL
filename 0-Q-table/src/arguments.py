import argparse


def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    parser.add_argument('--lr', metavar='L', type=float, default=0.8,
                        help='learning rate')
    parser.add_argument('--y', metavar='Y', type=float, default=0.95,
                        help='discount factor')
    parser.add_argument('--r', metavar='R', type=int, default=2000,
                        help='episode number (round)')
    parser.add_argument('--x', metavar='X', type=float, default=0.998,
                        help='e-greedy factor')
    parser.add_argument('--s', metavar='S', type=int, default=100,
                        help='step number')
    parser.add_argument('--size', metavar='N', type=int, default=6,
                        help='map size (N x N)')

    parser.add_argument('--start', metavar='T', type=str, default="(0, 0)",
                        help='a start cordinate')
    parser.add_argument('--goals', metavar='G', type=str, default="{(4, 4)}",
                        help='set of goal cordinate(s)')
    parser.add_argument('--obs', metavar='O', type=str, default="{(3, 4), (4, 2), (4, 3), (4, 5)}",
                        help='set of obstacle cordinate(s)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    print(args)
