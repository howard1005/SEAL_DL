# -*- coding: utf-8 -*-
import sys
import pdb
import time

class MNIST_DL:
  def __init__(self):
    self.requset = {}
    self.response = {}

  def check_img(self, mnist, mnist_idx):
    import numpy as np
    import matplotlib.pyplot as plt

    print("훈련 이미지 :", mnist.train.images.shape)
    print("훈련 라벨:", mnist.train.labels.shape)
    print("테스트 이미지 : ", mnist.test.images.shape)
    print("테스트 라벨 : ", mnist.test.labels.shape)
    print("검증 이미지 : ", mnist.validation.images.shape)
    print("검증 라벨 : ", mnist.validation.labels.shape)
    print('\n')

    print('[label]')
    print('one-hot vector label = ', mnist.train.labels[mnist_idx])
    print('number label = ', np.argmax(mnist.train.labels[mnist_idx]))
    print('\n')

    print('[image]')
    '''
    for index, pixel in enumerate(mnist.train.images[mnist_idx]):
      if index % 28 == 0:
        print('\n')
      else:
        print("%10f" % pixel, end="")
    print('\n')
    '''

    plt.figure(figsize=(5, 5))
    image = np.reshape(mnist.train.images[mnist_idx], [28, 28])
    plt.imshow(image, cmap='Greys')
    plt.show()

  def MNIST_train(self):
    # MNIST 데이터를 다운로드 한다.
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    # import tensorflow_datasets as tfds
    # mnist = tfds.load('mnist', split='train', shuffle_files=True, download=False)

    self.check_img(mnist, 0)
    # pdb.set_trace()

    # TensorFlow 라이브러리를 추가한다.
    import tensorflow as tf

    # 변수들을 설정한다.
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    W = tf.Variable(dtype=tf.float32, initial_value=tf.zeros([784, 10]))
    b = tf.Variable(dtype=tf.float32, initial_value=tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # cross-entropy 모델을 설정한다.
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 경사하강법으로 모델을 학습한다.
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
      print("i : {}".format(i))
      batch_xs, batch_ys = mnist.train.next_batch(550)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 학습된 모델이 얼마나 정확한지를 출력한다.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    start_time = time.time()
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print("time : {}".format(time.time() - start_time))

    # 모델 저장
    saver = tf.train.Saver()
    saver.save(sess, './save_model/mnist', global_step=0, write_meta_graph=False)
    # (모델의 session, 저장할 이름과 경로, 몇번째 학습한 것을 저장할 지, meta graph를 생성할지)
    #saver.restore(sess, tf.train.latest_checkpoint('./save_model'))

    y_np = sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images})
    yhat_np = sess.run(tf.argmax(y_, 1), feed_dict={y_: mnist.test.labels})
    W_np = W.eval(session=sess)
    b_np = b.eval(session=sess)

    with open('y.txt', 'w') as f:
      for n in y_np.flatten():
        f.write(str(n) + ',')

    with open('yhat.txt', 'w') as f:
      for n in yhat_np.flatten():
        f.write(str(n) + ',')

    with open('w.txt', 'w') as f:
      for n in W_np.flatten():
        f.write(str(n) + ',')

    with open('b.txt', 'w') as f:
      for n in b_np.flatten():
        f.write(str(n) + ',')
    '''
    print("\nW_np\n")
    for i, n in enumerate(W_np):
      print("{} : {}".format(i, n))
    print("\nb_np\n")
    print(b_np)
    '''

    pdb.set_trace()

  def client_result(self, data):
    data_list = list(data)
    import operator
    index, value = max(enumerate(data_list), key=operator.itemgetter(1))
    print("index : {} value : {}".format(index, value))

#  def server_proc(self, data):


if __name__=="__main__":
  mnist_dl = MNIST_DL()

  cmd = 'MNIST_train' # default cmd
  data = None

  if len(sys.argv) >= 2:
    cmd = sys.argv[1]
    data = sys.argv[2]
  else:
    print("invaild arg -> default cmd : {}".format(cmd))

  print("cmd : {}".format(cmd))
  if cmd == 'MNIST_train':
    mnist_dl.MNIST_train()
  elif cmd == 'client_result':
    mnist_dl.client_result(data)
