from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TestLogger

def build_callbacks(env_name, filename_pre, filename_exp):
    checkpoint_weights_filename = filename_pre + env_name + '_weights_{step}.h5f'
    log_filename = filename_pre+ filename_exp + '/ddpg_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(
        checkpoint_weights_filename, interval=50000)]
    callbacks += [FileLogger(log_filename, interval=50000)]

    return callbacks


def save_process_noise(env_name, filename_pre, filename_exp, noise_insert, theta_insert):
    log_filename = filename_pre+ filename_exp + '/ddpg_{}_log.json'.format(env_name)
    with open(log_filename, 'rb+') as f:  # -1 offset backwards for 1 character, 2 pstart at the end of file
        f.seek(-1, 2)
        f.truncate()
    with open(log_filename, 'a') as f:
        f.write(',\"process_noise_std\":[{}],\"theta\":[{}]}}'.format(noise_insert, theta_insert))
