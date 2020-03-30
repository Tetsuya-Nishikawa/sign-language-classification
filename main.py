from Model import Model
import numpy as np
import tensorflow as tf
import io_data
import MyLibrary 

gpus = tf.config.experimental.list_physical_devices('GPU')
#GPUのメモリを制限する。
if gpus:
  try:
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
print("GPUのメモリを制限する")
def tensor_cast(inputs, labels):
    inputs = tf.cast(inputs, tf.float32)/255.0
    labels = labels 
    #inputs = tf.reshape(inputs, [1, inputs.shape[0], inputs.shape[1], inputs.shape[2]])
    return inputs, tf.cast(labels, tf.int64)


#if __name__ == '__main__':
def main():
    alpha = [10**(-5)]
    lambd = [3]
    drop_rate = [0.3]#0.5
    hparam_list = MyLibrary.make_hparam_list(alpha, lambd, drop_rate)
    #batch_size = 8
    batch_size = 10
    epochs = 30
    
    #グリッドリサーチ
    print(hparam_list)
    for hparm_key in hparam_list:
        np.random.seed(1234)
        tf.random.set_seed(seed=1234)
        print("hparam_listの", hparm_key, "番目:")
        model = Model("Adam", hparam_list[hparm_key], batch_size, epochs)
        train_dataset, test_dataset = io_data.read_dataset(batch_size)
        train_ds =   model.mirrored_strategy.experimental_distribute_dataset(train_dataset)
        test_ds  =   model.mirrored_strategy.experimental_distribute_dataset(test_dataset)
        
        model.train(train_ds, test_ds)
