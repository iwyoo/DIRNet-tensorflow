import tensorflow as tf
from WarpST import WarpST
from ops import *

class CNN(object):
  def __init__(self, name, is_train):
    self.name = name
    self.is_train = is_train
    self.reuse = None
  
  def __call__(self, x):
    with tf.variable_scope(self.name, reuse=self.reuse):
      x = conv2d(x, "conv1", 64, 3, 1, 
        "SAME", True, tf.nn.elu, self.is_train)
      x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], "SAME")

      x = conv2d(x, "conv2", 128, 3, 1, 
        "SAME", True, tf.nn.elu, self.is_train)
      x = conv2d(x, "out1", 128, 3, 1, 
        "SAME", True, tf.nn.elu, self.is_train)
      x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], "SAME")
      x = conv2d(x, "out2", 2, 3, 1, 
        "SAME", False, None, self.is_train)

    if self.reuse is None:
      self.var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.saver = tf.train.Saver(self.var_list)
      self.reuse = True

    return x

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)

class DIRNet(object):
  def __init__(self, sess, config, name, is_train):
    self.sess = sess
    self.name = name
    self.is_train = is_train

    # moving / fixed images
    im_shape = [config.batch_size] + config.im_size + [1]
    self.x = tf.placeholder(tf.float32, im_shape)
    self.y = tf.placeholder(tf.float32, im_shape)
    self.xy = tf.concat([self.x, self.y], 3)

    self.vCNN = CNN("vector_CNN", is_train=self.is_train)

    # vector map & moved image
    self.v = self.vCNN(self.xy)
    self.z = WarpST(self.x, self.v, config.im_size)

    if self.is_train :
      self.loss = ncc(self.y, self.z)
      #self.loss = mse(self.y, self.z)

      self.optim = tf.train.AdamOptimizer(config.lr)
      self.train = self.optim.minimize(
        - self.loss, var_list=self.vCNN.var_list)

    #self.sess.run(
    #  tf.variables_initializer(self.vCNN.var_list))
    self.sess.run(
      tf.global_variables_initializer())

  def fit(self, batch_x, batch_y):
    _, loss = \
      self.sess.run([self.train, self.loss], 
      {self.x:batch_x, self.y:batch_y})
    return loss

  def deploy(self, dir_path, x, y):
    z = self.sess.run(self.z, {self.x:x, self.y:y})
    for i in range(z.shape[0]):
      save_image_with_scale(dir_path+"/{:02d}_x.tif".format(i+1), x[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_y.tif".format(i+1), y[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_z.tif".format(i+1), z[i,:,:,0])

  def save(self, dir_path):
    self.vCNN.save(self.sess, dir_path+"/model.ckpt")

  def restore(self, dir_path):
    self.vCNN.restore(self.sess, dir_path+"/model.ckpt")
