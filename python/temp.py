import pickle as pkl

from create_config import create_configs, create_slurm_jobs

if __name__ == '__main__':
    r = [0.1]
    s = [5]
    h = [0.01, 0.05, 0.1, 0.2]
    mu = [0.01, 0.02, 0.03, 0.05, 0.1]
    ls_reg = [1e-6]
    create_configs(checkpoint_base_dir='~/training/checkpoints', log_base_dir='~/training/logs',
                   r=r, s=s, h=h, mu=mu, ls_regularizer=ls_reg)

    # user_mail = 'klug.nikolas@gmail.com'
    # script = '/homes/stud/klugniko/thesis-mechanical-regression/python/run.py'
    # config_base_dir = '/homes/stud/klugniko/configs/20_12_04'
    # config_dir = '/home/nikolas/Projects/thesis-mechanical-regression/training/configs/20_12_04'
    # out_dir = '/home/nikolas/Projects/thesis-mechanical-regression/training/slurm_jobs/20_12_04'
    #
    # create_slurm_jobs(config_dir, out_dir, user_mail, script, config_base_dir)
