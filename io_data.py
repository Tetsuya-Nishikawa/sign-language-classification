import tensorflow as tf
import skvideo.io
import imutils
import glob
import os

#input_path = "/home/ubuntu/nfs/cybozu/Temporal_Transformation/SampledVideo"
#/home/ubuntu/nfs/cybozu/video_dataset/train_dataset
#/home/ubuntu/nfs/cybozu/video_dataset/test_dataset
train_path = "/home/ubuntu/nfs/cybozu/video_dataset63/tfrecords/train.tfrecords"
test_path  = "/home/ubuntu/nfs/cybozu/video_dataset63/tfrecords/test.tfrecords"

def tensor_cast(inputs, labels):
    inputs = tf.cast(inputs, tf.float32)/255.0
    labels = labels - 1
    #labels = tf.one_hot(labels, 64)
    #labels = tf.reshape(labels, [-1,])
    return inputs, tf.cast(labels, tf.int64)

def GenerateMaskTensor(video):
    timesteps = 201
    l = [False]*timesteps
    frames = video.shape[0]
    for index in range(frames):
        l[index] = True

    return l

#tfrecordの処理の参考URL
#https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(video, label, video_shape):
    feature = {
        'video': _bytes_feature(video),
        'label': _int64_feature(label),
        'len_seq':_int64_feature(video_shape[0]),
        'height': _int64_feature(video_shape[1]),
        'width': _int64_feature(video_shape[2]),
        'depth': _int64_feature(video_shape[3]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_tfrecord(serialized_example):
    feature_description = {
        'video' : tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'len_seq':tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    video = tf.io.parse_tensor(example['video'], out_type = tf.uint8)
    video_shape = [example['len_seq'], example['height'], example['width'], example['depth']]
    video = tf.reshape(video, video_shape)

    return video, example['label']

def write_dataset():
    train_input_path = "/home/ubuntu/nfs/cybozu/video_dataset30/train_dataset/*.mp4"
    test_input_path = "/home/ubuntu/nfs/cybozu/video_dataset30/test_dataset/*.mp4"

    with tf.io.TFRecordWriter(train_path) as writer:

        video_list = glob.glob(train_input_path)

        for video_path in video_list:
            video = skvideo.io.vread(video_path)

            video_bytes = tf.io.serialize_tensor(video)

            label = video_path.split(os.sep)[7][0:-12]
            label = int(label)
            print("現在処理しているビデオのインデックスは、", video_list.index(video_path))
            example = serialize_example(video_bytes, label, video.shape)
            writer.write(example)
    with tf.io.TFRecordWriter(test_path) as writer:

        video_list = glob.glob(test_input_path)

        for video_path in video_list:
            video = skvideo.io.vread(video_path)

            video_bytes = tf.io.serialize_tensor(video)

            label = video_path.split(os.sep)[7][0:-12]
            label = int(label)
            print("現在処理しているビデオのインデックスは、", video_list.index(video_path))
            example = serialize_example(video_bytes, label, video.shape)
            writer.write(example)


def read_dataset(BATCH_SIZE):
    tfrecord_train_dataset = tf.data.TFRecordDataset(train_path)
    parsed_train_dataset = tfrecord_train_dataset.map(parse_tfrecord)
    tfrecord_test_dataset = tf.data.TFRecordDataset(test_path)
    parsed_test_dataset = tfrecord_test_dataset.map(parse_tfrecord)

    train_dataset = parsed_train_dataset.map(tensor_cast).shuffle(buffer_size=100, seed=100).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset =  parsed_test_dataset.map(tensor_cast).shuffle(buffer_size=100,  seed=100).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, test_dataset

if __name__ == '__main__':
    write_dataset()
