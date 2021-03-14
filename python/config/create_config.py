import argparse
import json
from datetime import datetime
from glob import glob
from itertools import product
from os import path


def create_slurm_jobs(local_config_dir, output_dir, user_mail, script_location, config_base_dir, job_name):
    with open('templates/default_template.slurm.txt', 'r') as file:
        template = file.read()
    all_configs = glob(path.join(local_config_dir, '*.json'))
    script_dir = path.dirname(script_location)

    commands = []
    for config in all_configs:
        job_name = path.basename(config).replace('.json', '')
        remote_config_path = path.join(config_base_dir, path.basename(config))
        command = 'python3 ' + script_location + ' ' + remote_config_path + ' > out/' + job_name + '.out &\n'

        commands.append(command)
    commands.append('wait')
    filled_template = template.format(mem=str(2 * len(all_configs)), job_name=job_name, user_mail=user_mail,
                                      scripts=''.join(commands),
                                      script_dir=script_dir)
    with open(path.join(output_dir, job_name + '.slurm'), 'w') as out_file:
        out_file.write(filled_template)


def create_configs(s, r, h, mu, ls_regularizer, loss, log_base_dir, checkpoint_base_dir, dataset):
    for s1, r1, h1, mu1, ls_regularizer1, loss1, dataset1 in product(s, r, h, mu, ls_regularizer, loss, dataset):
        name = "{}_s_{}_r_{}_h_{}_mu_{}_reg_{}".format(dataset1, s1, r1, h1, mu1, ls_regularizer1)
        create_config(name, s=s1, r=r1, h=h1, mu=mu1, ls_reg=ls_regularizer1, loss=loss1, log_base_dir=log_base_dir,
                      checkpoint_base_dir=checkpoint_base_dir, dataset=dataset1)


def create_config(config_name, log_base_dir='', checkpoint_base_dir='', s=5, r=0.1, h=0, mu=0, ls_reg=1e-6,
                  loss='optimal_recovery', dataset='default_swiss_roll'):
    config_name = datetime.now().strftime('%Y_%m_%d_') + str(config_name)
    config = {
        "name": config_name,
        "log_base_dir": log_base_dir,
        "checkpoint_base_dir": checkpoint_base_dir,
        "dataset": dataset,
        "s": s,
        "r": r,
        "h": h,
        "mu": mu,
        "ls_regularizer": ls_reg,
        "loss": loss
    }
    with open(path.join('../training/configs', config_name + '.json'), 'w') as file:
        json.dump(config, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config_name', help=' name of the config to be created.')
    args = parser.parse_args()

    create_config(args.config_name)
