import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='Process CSV downloaded')
    # parser.add_argument('config', help='train config file path')
    parser.add_argument('file', help='csv file')
    parser.add_argument('-o', '--out', type=str, help='output file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    file = args.file
    out_file = args.out
    if out_file is None:
        out_file = file.replace('.csv', 'converted.csv')

    # config_set = []
    # for root, dirs, files in os.walk(args.config):
    #     config_set.extend(list(files))
    # config_set = set(config_set)
    #
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        out_dict = {}
        for row in reader:
            name = row[0].split('.')[0].replace('-davis', '')
            out_dict[name] = row[1:]
