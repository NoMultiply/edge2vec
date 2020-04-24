import os
import tensorflow as tf
import numpy

directory="Epinions-500"
size = 1000

def convert_to(name, len):
  file_input = os.path.join(directory, name + '.csv')
  file_output = os.path.join(directory, name + '.tfrecords')
  
  f_input = open(file_input)
  print('Writing', file_output)
  writer = tf.python_io.TFRecordWriter(file_output)

  k = 0
  for line in f_input:
    values = line.split(",")
    fs = numpy.array(map(float, values[0:len]), dtype = numpy.float32)
    ms = numpy.array(map(int, values[len:2*len]), dtype = numpy.uint8)
    example = tf.train.Example(features=tf.train.Features(feature={
      'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fs.tostring()])),
      'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms.tostring()]))}))
    writer.write(example.SerializeToString())
    k=k+1
    if k % 1000 == 0:
      print(k)
  writer.close()
  f_input.close()

def convert_to2(name, len):
  file_input = os.path.join(directory, name + '.csv')
  file_output = os.path.join(directory, name + '.tfrecords')
  
  f_input = open(file_input)
  print('Writing', file_output)
  writer = tf.python_io.TFRecordWriter(file_output)

  k = 0
  for line in f_input:
    values = line.split(",")
    fs=[]
    ms=[]
    for i in xrange(7):
      fs.append(numpy.array(map(float, values[2*i*len:(2*i+1)*len]), dtype = numpy.float32))
      ms.append(numpy.array(map(int, values[(2*i+1)*len:(2*i+2)*len]), dtype = numpy.uint8))
    example = tf.train.Example(features=tf.train.Features(feature={
      'example': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fs[0].tostring()])),
      'example_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms[0].tostring()])),

      'positve': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fs[1].tostring()])),
      'positive_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms[1].tostring()])),

      'negtive1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fs[2].tostring()])),
      'negtive1_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms[2].tostring()])),

      'negtive2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fs[3].tostring()])),
      'negtive2_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms[3].tostring()])),

      'negtive3': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fs[4].tostring()])),
      'negtive3_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms[4].tostring()])),

      'negtive4': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fs[5].tostring()])),
      'negtive4_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms[5].tostring()])),

      'negtive5': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fs[6].tostring()])),
      'negtive5_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms[6].tostring()]))
      }))
    writer.write(example.SerializeToString())
    k=k+1
    if k % 1000 == 0:
      print(k)
  writer.close()
  f_input.close()

def main(argv):
  convert_to2("negtive", size)
  # Convert to Examples and write the result to TFRecords.
  convert_to('data', size)
  convert_to('train', size) 
  convert_to('test', size)

  

if __name__ == '__main__':
  tf.app.run()