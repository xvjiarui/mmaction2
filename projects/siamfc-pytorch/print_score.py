import mmcv
import argparse
import numpy as np
from terminaltables import AsciiTable

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('file', help='train config file path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    report = mmcv.load(args.file)
    for tracker in report:
        print(tracker)
        class_table_data = [['video', 'success_score', 'precision_score', 'success_rate', 'speed_fps']]
        for video in report[tracker]['seq_wise']:
            video_seq = report[tracker]['seq_wise'][video]
            success_score = video_seq['success_score'] * 100
            success_score = np.round(success_score, 2)
            precision_score = video_seq['precision_score'] * 100
            precision_score = np.round(precision_score, 2)
            success_rate = video_seq['success_rate'] * 100
            success_rate = np.round(success_rate, 2)
            speed_fps = video_seq['speed_fps']
            speed_fps = np.round(speed_fps, 2)
            class_table_data.append([video, success_score, precision_score, success_rate, speed_fps])
        video_overall = report[tracker]['overall']
        success_score = video_overall['success_score'] * 100
        success_score = np.round(success_score, 2)
        precision_score = video_overall['precision_score'] * 100
        precision_score = np.round(precision_score, 2)
        success_rate = video_overall['success_rate'] * 100
        success_rate = np.round(success_rate, 2)
        speed_fps = video_overall['speed_fps']
        speed_fps = np.round(speed_fps, 2)
        class_table_data.append(
            ['overall', success_score, precision_score, success_rate, speed_fps])
        table = AsciiTable(class_table_data)
        print(table.table)


if __name__ == '__main__':
    main()
