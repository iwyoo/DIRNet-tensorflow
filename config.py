class Config(object):
  pass

def get_config(is_train):
  config = Config()
  if is_train:
    config.batch_size = 64
    config.im_size = [28, 28]
    config.lr = 1e-4
    config.iteration = 10000

    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
  else:
    config.batch_size = 10
    config.im_size = [28, 28]

    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
  return config
