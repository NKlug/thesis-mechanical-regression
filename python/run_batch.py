import argparse
from glob import glob
from os import path, makedirs
from subprocess import Popen, STDOUT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config_dir', help=' location of the configs for the current training.')
    args = parser.parse_args()

    for config_path in glob(path.join(args.config_dir, '*.json')):
        print('Running with ' + config_path)
        output_file_name = path.basename(config_path).replace('.json', '') + '.out'
        output_file = path.realpath(path.expanduser(
            path.join('~/training/outputs', path.basename(args.config_dir), output_file_name)))
        makedirs(path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            Popen(['python3', 'run.py', config_path], stdout=out_file, stderr=STDOUT)
