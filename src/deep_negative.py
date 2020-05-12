from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from src.deep_flags import setup_flags


class AutoEncoder(object):
    _weights_str = "weights{0}"
    _biases_str = "biases{0}"

    def __init__(self, shape, sess):
        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
        self.__num_hidden_layers = len(self.__shape) - 2

        self.__variables = {}
        self.__sess = sess

        self._setup_variables()

    @property
    def shape(self):
        return self.__shape

    @property
    def num_hidden_layers(self):
        return self.__num_hidden_layers

    @property
    def session(self):
        return self.__sess

    def __getitem__(self, item):
        return self.__variables[item]

    def __setitem__(self, key, value):
        self.__variables[key] = value

    def _setup_variables(self):
        with tf.name_scope("autoencoder_variables"):
            for i in range(self.__num_hidden_layers):
                # Train weights
                name_w = self._weights_str.format(i + 1)
                w_shape = (self.__shape[i], self.__shape[i + 1])
                a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
                w_init = tf.random_uniform(w_shape, -1 * a, a)
                self[name_w] = tf.Variable(w_init,
                                           name=name_w,
                                           trainable=True)
                # Train biases
                name_b = self._biases_str.format(i + 1)
                b_shape = (self.__shape[i + 1],)
                b_init = tf.zeros(b_shape)
                self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

                if i < self.__num_hidden_layers:
                    # Hidden layer fixed weights (after pretraining before fine tuning)
                    self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                                                          name=name_w + "_fixed",
                                                          trainable=False)

                    # Hidden layer fixed biases
                    self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                                                          name=name_b + "_fixed",
                                                          trainable=False)

                    # Pretraining output training biases
                    name_b_out = self._biases_str.format(i + 1) + "_out"
                    b_shape = (self.__shape[i],)
                    b_init = tf.zeros(b_shape)
                    self[name_b_out] = tf.Variable(b_init,
                                                   trainable=True,
                                                   name=name_b_out)

    def _w(self, n, suffix=""):
        return self[self._weights_str.format(n) + suffix]

    def _b(self, n, suffix=""):
        return self[self._biases_str.format(n) + suffix]

    def get_variables_to_init(self, n):
        assert n > 0
        assert n <= self.__num_hidden_layers

        vars_to_init = [self._w(n), self._b(n)]

        if n <= self.__num_hidden_layers:
            vars_to_init.append(self._b(n, "_out"))

        if 1 < n <= self.__num_hidden_layers:
            vars_to_init.append(self._w(n - 1, "_fixed"))
            vars_to_init.append(self._b(n - 1, "_fixed"))

        return vars_to_init

    @staticmethod
    def _activate(x, w, b, transpose_w=False):
        y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
        return y

    def pretrain_net(self, input_pl, n, is_target=False):
        assert n > 0
        assert n <= self.__num_hidden_layers

        last_output = input_pl
        for i in range(n - 1):
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")

            last_output = self._activate(last_output, w, b)

        if is_target:
            return last_output

        # last_output = tf.nn.dropout(last_output, 0.8)
        last_output = self._activate(last_output, self._w(n), self._b(n))

        out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                             transpose_w=True)
        out = tf.maximum(out, 1.e-9)
        out = tf.minimum(out, 1 - 1.e-9)
        return out

    def reconstuction_net(self, p_net, n):
        r_net = p_net
        for i in range(n - 1):
            j = n - 1 - i
            r_net = self._activate(r_net, self._w(j), self._b(j, "_out"),
                                   transpose_w=True)
        return r_net

    def transform_net(self, input_pl):
        last_output = input_pl
        for i in range(self.__num_hidden_layers):
            w = self._w(i + 1)
            b = self._b(i + 1)

            last_output = self._activate(last_output, w, b)
            last_output = tf.maximum(last_output, 1.e-9)
            last_output = tf.minimum(last_output, 1 - 1.e-9)

        return last_output

    def finetune_reconstruction_net(self, t_net):
        last_output = t_net
        for i in range(self.__num_hidden_layers):
            j = self.__num_hidden_layers - i
            w = self._w(j)
            b = self._b(j, "_out")

            last_output = self._activate(last_output, w, b, transpose_w=True)
            last_output = tf.maximum(last_output, 1.e-9)
            last_output = tf.minimum(last_output, 1 - 1.e-9)

        return last_output


def read_my_file_format(filename_queue, n_input):
    reader = tf.TFRecordReader()
    _, record_string = reader.read(filename_queue)
    features = tf.parse_single_example(
        record_string,
        features={
            'features': tf.FixedLenFeature([], tf.string),
            'mask': tf.FixedLenFeature([], tf.string),
        })
    record = tf.decode_raw(features['features'], tf.float32)
    mask = tf.decode_raw(features['mask'], tf.uint8)
    mask = tf.cast(mask, tf.float32)
    record.set_shape([n_input])
    mask.set_shape([n_input])
    return record, mask


def read_my_file_format2(filename_queue, n_input):
    reader = tf.TFRecordReader()
    _, record_string = reader.read(filename_queue)
    features = tf.parse_single_example(
        record_string,
        features={
            'example': tf.FixedLenFeature([], tf.string),
            'positve': tf.FixedLenFeature([], tf.string),
            'negtive1': tf.FixedLenFeature([], tf.string),
            'negtive2': tf.FixedLenFeature([], tf.string),
            'negtive3': tf.FixedLenFeature([], tf.string),
            'negtive4': tf.FixedLenFeature([], tf.string),
            'negtive5': tf.FixedLenFeature([], tf.string),
            'example_mask': tf.FixedLenFeature([], tf.string),
            'positive_mask': tf.FixedLenFeature([], tf.string),
            'negtive1_mask': tf.FixedLenFeature([], tf.string),
            'negtive2_mask': tf.FixedLenFeature([], tf.string),
            'negtive3_mask': tf.FixedLenFeature([], tf.string),
            'negtive4_mask': tf.FixedLenFeature([], tf.string),
            'negtive5_mask': tf.FixedLenFeature([], tf.string),
        })
    example = tf.decode_raw(features['example'], tf.float32)
    positve = tf.decode_raw(features['positve'], tf.float32)
    negtive1 = tf.decode_raw(features['negtive1'], tf.float32)
    negtive2 = tf.decode_raw(features['negtive2'], tf.float32)
    negtive3 = tf.decode_raw(features['negtive3'], tf.float32)
    negtive4 = tf.decode_raw(features['negtive4'], tf.float32)
    negtive5 = tf.decode_raw(features['negtive5'], tf.float32)

    example_mask = tf.decode_raw(features['example_mask'], tf.uint8)
    positve_mask = tf.decode_raw(features['positive_mask'], tf.uint8)
    negtive1_mask = tf.decode_raw(features['negtive1_mask'], tf.uint8)
    negtive2_mask = tf.decode_raw(features['negtive2_mask'], tf.uint8)
    negtive3_mask = tf.decode_raw(features['negtive3_mask'], tf.uint8)
    negtive4_mask = tf.decode_raw(features['negtive4_mask'], tf.uint8)
    negtive5_mask = tf.decode_raw(features['negtive5_mask'], tf.uint8)

    example_mask = tf.cast(example_mask, tf.float32)
    positve_mask = tf.cast(positve_mask, tf.float32)
    negtive1_mask = tf.cast(negtive1_mask, tf.float32)
    negtive2_mask = tf.cast(negtive2_mask, tf.float32)
    negtive3_mask = tf.cast(negtive3_mask, tf.float32)
    negtive4_mask = tf.cast(negtive4_mask, tf.float32)
    negtive5_mask = tf.cast(negtive5_mask, tf.float32)

    example.set_shape([n_input])
    positve.set_shape([n_input])
    negtive1.set_shape([n_input])
    negtive2.set_shape([n_input])
    negtive3.set_shape([n_input])
    negtive4.set_shape([n_input])
    negtive5.set_shape([n_input])
    example_mask.set_shape([n_input])
    positve_mask.set_shape([n_input])
    negtive1_mask.set_shape([n_input])
    negtive2_mask.set_shape([n_input])
    negtive3_mask.set_shape([n_input])
    negtive4_mask.set_shape([n_input])
    negtive5_mask.set_shape([n_input])
    return example, positve, negtive1, negtive2, negtive3, negtive4, negtive5, example_mask, positve_mask, negtive1_mask, negtive2_mask, negtive3_mask, negtive4_mask, negtive5_mask


def input_pipeline(filenames, batch_size, n_input, num_epochs=None, infer_mode=False):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
    example = read_my_file_format(filename_queue, n_input)
    capacity = 5 * batch_size
    if (infer_mode):
        num_threads = 1
    else:
        num_threads = 8
        example_batch, mask_batch = tf.train.batch(
            example, batch_size=batch_size, capacity=capacity, num_threads=num_threads,
            allow_smaller_final_batch=infer_mode)
        return example_batch, mask_batch


def input_pipeline2(filenames, batch_size, n_input):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=False)
    example = read_my_file_format2(filename_queue, n_input)
    capacity = 5 * batch_size
    num_threads = 8
    example_batch = tf.train.batch(
        example, batch_size=batch_size, capacity=capacity, num_threads=num_threads)
    return example_batch


def training(loss, learning_rate):
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op


# def loss_x_entropy(output, target):
#  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))

def loss_rsme(x, reconstruction, mask=None):
    if mask is not None:
        return 0.5 * tf.reduce_sum(tf.pow(tf.multiply(tf.subtract(reconstruction, x), mask), 2.0))
    return 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction, x), 2.0))


def loss_negtive(x, y, positve):
    if positve is True:
        # return tf.reduce_sum(-tf.log(tf.sigmoid(tf.reduce_sum(tf.multiply(x, y), 1))))
        y = tf.sigmoid(tf.reduce_sum(tf.multiply(x, y), 1))
        return tf.reduce_sum(-tf.log(
            tf.clip_by_value(y, 1e-8, tf.reduce_max(y))
        ))
    else:
        # return tf.reduce_sum(-tf.log(tf.sigmoid(tf.reduce_sum(-tf.multiply(x, y), 1))))
        y = tf.sigmoid(tf.reduce_sum(-tf.multiply(x, y), 1))
        return tf.reduce_sum(-tf.log(
            tf.clip_by_value(y, 1e-8, tf.reduce_max(y))
        ))


def output_file2(f, embedding, n_hidden, label):
    str_list = [str(label) + ' ']
    for i in range(n_hidden):
        str_list.append(str(i + 1) + ':' + str(embedding[i]) + ' ')
    f.write(''.join(str_list))
    f.write('\n')


def output_file(f, embedding, n_hidden, label):
    str_list = []
    for i in range(n_hidden):
        str_list.append(str(embedding[i]))
    f.write(' '.join(str_list))
    f.write('\n')


def pretrain(image_pixels, directory):
    FLAGS = setup_flags(image_pixels, directory)
    with tf.Graph().as_default() as g:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        directory = FLAGS.directory
        num_hidden = FLAGS.num_hidden_layers
        batch_size = FLAGS.batch_size
        num_data = int(FLAGS.num_data / batch_size)
        num_train = FLAGS.num_train
        num_test = FLAGS.num_test
        ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                            for j in range(num_hidden)]
        ae_shape = [FLAGS.image_pixels] + ae_hidden_shapes + [FLAGS.num_classes]
        combine_epochs = FLAGS.combine_epochs
        pretraining_epochs = FLAGS.pretraining_epochs

        finetuning_epochs = FLAGS.finetuning_epochs
        learning_rate = FLAGS.finetune_learning_rate
        alpha = FLAGS.alpha
        num_finetune = int(FLAGS.num_finetune / batch_size)

        ae = AutoEncoder(ae_shape, sess)

        learning_rates = {j: getattr(FLAGS,
                                     "pre_layer{0}_learning_rate".format(j + 1))
                          for j in range(num_hidden)}

        input_, mask = input_pipeline([os.path.join(directory, "data.tfrecords")], batch_size=batch_size,
                                      n_input=FLAGS.image_pixels)
        record = input_pipeline2(
            [os.path.join(directory, "negative.tfrecords")], batch_size=batch_size, n_input=FLAGS.image_pixels)
        input_train, _ = input_pipeline([os.path.join(directory, "train.tfrecords")], batch_size=1,
                                        n_input=FLAGS.image_pixels)
        input_test, _ = input_pipeline([os.path.join(directory, "test.tfrecords")], batch_size=1,
                                       n_input=FLAGS.image_pixels)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for i in range(len(ae_shape) - 2):
                n = i + 1
                with tf.variable_scope("pretrain_{0}".format(n)):
                    target_ = input_
                    layer = ae.pretrain_net(input_, n)

                    with tf.name_scope("target"):
                        target_for_loss = ae.pretrain_net(target_, n, is_target=True)
                        if n == 1:
                            loss = loss_rsme(layer, target_for_loss, mask)
                        else:
                            loss = loss_rsme(layer, target_for_loss)

                    reconstruction = ae.reconstuction_net(layer, n)
                    global_loss = loss_rsme(reconstruction, input_, mask)

                    variables_before = tf.all_variables()
                    train_op = training(loss, learning_rates[i])
                    variables_after = tf.all_variables()

                    vars_to_init = ae.get_variables_to_init(n)
                    vars_to_init.extend([variable for variable in variables_after if variable not in variables_before])
                    sess.run(tf.initialize_variables(vars_to_init))

                    # sess.run(tf.initialize_all_variables())
                    print("\n\n")
                    print("| Training Step | Local Loss    | Global Loss   |  Layer  |   Epoch  |")
                    print("|---------------|---------------|---------------|---------|----------|")

                    # for step in xrange(11):
                    for step in range(pretraining_epochs * num_data):

                        _, loss_value, global_loss_value = sess.run([train_op, loss, global_loss])

                        if step % 10 == 0:
                            output = "| {0:>13} | {1:13.4f} | {2:13.4f} | Layer {3} | Epoch {4}  |" \
                                .format(step, loss_value, global_loss_value, n, step // num_data + 1)

                            print(output)

            target_ = input_
            layer = ae.finetune_reconstruction_net(ae.transform_net(input_))
            combine_loss = loss_rsme(layer, target_, mask)

            variables_before = tf.all_variables()
            combine_op = training(combine_loss, learning_rates[0])
            variables_after = tf.all_variables()

            sess.run(
                tf.initialize_variables([variable for variable in variables_after if variable not in variables_before]))
            print("\n\n")
            print("| Combine Step    | Local Loss	   |   Epoch  |")
            print("|-----------------|---------------|----------|")
            for step in range(combine_epochs * num_data):

                _, loss_value = sess.run([combine_op, combine_loss])

                if step % 10 == 0:
                    output = "| {0:>13} | {1:13.4f} | Epoch {2}  |".format(step, loss_value, step // num_data + 1)

                    print(output)

            hidden = []
            reconstruction = []
            for i in range(7):
                hidden.append(ae.transform_net(record[i]))
                reconstruction.append(ae.finetune_reconstruction_net(hidden[i]))

            negtive_loss = loss_negtive(hidden[0], hidden[1], True)
            for k in range(5):
                # return tf.reduce_sum(-tf.log(tf.sigmoid(tf.reduce_sum(-tf.multiply(x, y), 1))))
                negtive_loss += loss_negtive(hidden[0], hidden[k + 2], False)

            # reconstruction_loss = loss_rsme(record[0], reconstruction[0])
            reconstruction_loss = loss_rsme(record[0], reconstruction[0], record[7])
            for k in range(6):
                reconstruction_loss += loss_rsme(record[k + 1], reconstruction[k + 1], record[k + 8])

            total_loss = alpha * negtive_loss + reconstruction_loss
            # total_loss = reconstruction_loss
            variables_before = tf.all_variables()
            finetuning_op = training(total_loss, learning_rate)
            variables_after = tf.all_variables()
            sess.run(tf.initialize_variables([variable
                                              for variable in variables_after if variable not in variables_before]))
            print("\n\n")
            print("| Finetuning Step | negative Loss  | reconst. Loss | total Loss    |   Epoch  |")
            print("|-----------------|---------------|---------------|---------------|----------|")
            for step in range(finetuning_epochs * num_finetune):

                _, n_loss, r_loss, t_loss, = sess.run([finetuning_op, negtive_loss,
                                                       reconstruction_loss, total_loss])

                if step % 10 == 0:
                    output = "| {0:>13} | {1:13.4f} | {2:13.4f} | {3:13.4f} | Epoch {4}  |".format(
                        step, n_loss, r_loss, t_loss, step // num_finetune + 1)

                    print(output)

            with open(os.path.join(directory, "train.log"), 'w') as f:
                output_ = ae.transform_net(input_train)
                for i in range(num_train):
                    embedding = sess.run(output_)
                    output_file(f, embedding[0], getattr(FLAGS, "hidden{0}_units".format(num_hidden)), 1 - i % 2)
            with open(os.path.join(directory, "test.log"), 'w') as f:
                output_ = ae.transform_net(input_test)
                for i in range(num_test):
                    embedding = sess.run(output_)
                    output_file(f, embedding[0], getattr(FLAGS, "hidden{0}_units".format(num_hidden)), 1 - i % 2)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        return ae

# if __name__ == '__main__':
#     ae = pretrain()
