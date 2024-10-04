import datetime
import glob
import os
import pickle
import re
from shutil import copy

import torch


class Logger(object):
    def __init__(self, args, params=None):
        self.ignore_dirs = ['./__pycache__', './*log', './.idea', './runs', 'tmp', 'results*', './checkpoints', './data/', 'tmp*']
        DIR = args.dir
        os.makedirs(DIR, exist_ok=True)
        exp_num = 0
        while True:
            exp_dir = os.path.join(DIR, '{}'.format(exp_num))
            try:
                os.makedirs(exp_dir, exist_ok=False)
                break
            except:
                exp_num += 1
        self.exp_dir = exp_dir

        self.log(args, params)

    def log(self, args, params=None):
        """log code"""
        code_dir = os.path.join(self.exp_dir, 'code')
        os.makedirs(code_dir, exist_ok=False)
        files = [p for p in glob.glob("./**/*.py", recursive=True) if os.path.isfile(p)]

        def check_ignore(input, ignore_dirs):
            match_flg = False
            for ignore_dir in ignore_dirs:
                match = re.findall(ignore_dir, input)
                if len(match) != 0:
                    match_flg = True
            return match_flg

        for f in files:
            if check_ignore(f, self.ignore_dirs):
                continue
            os.makedirs(os.path.dirname(os.path.join(code_dir, f)), exist_ok=True)
            if os.path.isfile(f):
                copy(f, os.path.join(code_dir, f))

        notename = os.path.join(self.exp_dir, 'log.txt')
        with open(notename, 'w') as note:
            content = ''

            """log timestamp"""
            now = datetime.datetime.now()
            content += (str(now) + '\n')

            """log args"""
            for arg in vars(args):
                content += (arg + ' = ' + str(getattr(args, arg)) + '\n')

            """log params"""
            if not params == None:
                for param_name in params:
                    content += (param_name + ' = ' + str(params[param_name]) + '\n')
            note.write(content)

    def log_csv(self, results):
        if results is not None:
            data_dir = os.path.join(self.exp_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            for res_name in results:
                res_path = os.path.join(data_dir, res_name + '.csv')
                results[res_name].to_csv(res_path, index=False)

    def log_pickle(self, results):
        if results is not None:
            data_dir = os.path.join(self.exp_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            for res_name in results:
                res_path = os.path.join(data_dir, res_name + '.pickle')
                self.save(res_path, results[res_name])

    def save(self, path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            print('{} saved'.format(path))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        assert type(val) in (int, float, torch.Tensor), f'Error value should be number. actual:{val}'
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        if self.count == 0:
            return ''
        else:
            return self.avg


class MetricsCollector(object):
    def __init__(self, keys=[], write_path=None):
        self.keys = ['Epoch'] + keys
        self.line = {key: AverageMeter() for key in self.keys}
        self.write_path = write_path
        self.epoch = None
        with open(self.write_path, 'w') as f:
            for i, key in enumerate(self.keys):
                f.write(key)
                if len(self.keys) - 1 != i:  # skip last columns
                    f.write(',')
            f.write('\n')
            f.flush()

    def append(self, epoch, key, value, n=1):
        assert key in self.keys, f'assert: {key} in {self.keys}'
        self.line['Epoch'].update(epoch, 1)
        self.line[key].update(value, n)
        return self.line[key].get_avg()

    def write(self):
        text_line = ''
        dict_line = {}
        for i, key in enumerate(self.keys):
            text_line += f'{self.line[key].get_avg()}'
            dict_line[key] = self.line[key].get_avg()
            if len(self.keys) - 1 != i:  # skip last columns
                text_line += ','

        with open(self.write_path, 'a') as f:
            f.write(text_line)
            f.write('\n')
            f.flush()
        for key in self.keys:
            self.line[key].reset()
        return dict_line


if __name__ == '__main__':
    bucket = MetricsCollector(['TrainRobustLoss', 'TestRobustLoss'], './metric.csv')
    for epoch in range(10):
        for _ in range(5):
            bucket.append(epoch, 'TrainRobustLoss', _ + epoch)
            if epoch % 5 == 0:
                print(bucket.append(epoch, 'TestRobustLoss', _ + epoch))
        print(bucket.write())