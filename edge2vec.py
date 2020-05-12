import os
import sys
from configargparse import ArgumentParser
from src.deep_pre import DeepPre
from src.csv2tf_neg import convert
from src.deep_negative import pretrain


def parse_arguments():
    parser = ArgumentParser(description='Arguments For edge2vec')

    group = parser.add_argument_group('Base Configs')
    group.add_argument('-i', '--input', help='path to the input graph file', type=str, required=True)
    # group.add_argument('-o', '--output', help='path to the output embedding file', type=str, required=True)
    group.add_argument('-m', '--model', help='the output directory of model files', type=str, required=True)
    group.add_argument('-n', '--num', help='the maximum num of the node', type=int, required=True)
    group.add_argument('-s', '--sample', help='the num of negative samples', type=int, required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    sys.argv = sys.argv[:1]
    if not os.path.exists(args.model):
        os.makedirs(args.model)

    pre = DeepPre(args.input, args.model, args.num, args.num, args.sample)
    pre.read_data()
    pre.calculate()
    pre.write_csv()

    convert(args.model, args.num * 2)
    pretrain(args.num * 2, args.model)


if __name__ == '__main__':
    main()
