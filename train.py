import tensorflow as tf
from models import DIRNet
from config import get_config
from data import MNISTDataHandler
from ops import mkdir

def main():
  sess = tf.Session()
  config = get_config(is_train=True)
  mkdir(config.tmp_dir)
  mkdir(config.ckpt_dir)

  reg = DIRNet(sess, config, "DIRNet", is_train=True)
  dh = MNISTDataHandler("MNIST_data", is_train=True)

  for i in range(config.iteration):
    batch_x, batch_y = dh.sample_pair(config.batch_size)
    loss = reg.fit(batch_x, batch_y)
    print("iter {:>6d} : {}".format(i+1, loss))

    if (i+1) % 1000 == 0:
      reg.deploy(config.tmp_dir, batch_x, batch_y)
      reg.save(config.ckpt_dir)

if __name__ == "__main__":
  main()
