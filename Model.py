import tensorflow as tf
import numpy as np
import os
import sys
import MyLayer
import MyLibrary

class Model(tf.keras.Model):
    def __init__(self, opt_name, hp, batch_size, epochs):
        super(Model, self).__init__()
        a = hp["alpha"]
        l = hp["lambd"]
        d = hp["drop_rate"]
        print("alpha:", a, "l", l, "drop_rate", d)
        self.lambd = l
        self.alpha = a
        self.drop_rate = d
        self.batch_size = batch_size
        self.epochs = epochs
        self.y_list = []#学習データに対する認識精度のリスト
        self.t_list  = []#検証データに対する認識精度のリスト
        
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        
        with self.mirrored_strategy.scope():
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
            self.train_acc =  tf.keras.metrics.SparseCategoricalAccuracy()
            self.test_acc  =  tf.keras.metrics.SparseCategoricalAccuracy()
        self.bn1         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), input_shape=(112, 200, 3))
        self.conv1   = tf.keras.layers.Conv3D(64, 3, padding="same")
        self.pool1   = tf.keras.layers.AveragePooling3D()

        self.bn2         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv2   = tf.keras.layers.Conv3D(128, 3, padding="same")
        self.pool2   = tf.keras.layers.AveragePooling3D()

        self.bn3         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv3   = tf.keras.layers.Conv3D(256, 3, padding="same")

        self.bn4         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv4   = tf.keras.layers.Conv3D(256, 3, padding="same")
        self.pool4   = tf.keras.layers.AveragePooling3D()

        self.bn5         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv5   = tf.keras.layers.Conv3D(512, 3, padding="same")
 
        self.bn6         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv6   = tf.keras.layers.Conv3D(512, 3, padding="same")
        self.pool6   = tf.keras.layers.AveragePooling3D()
 
        self.bn7         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv7   = tf.keras.layers.Conv3D(512, 3, padding="same")

        self.bn8         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv8   = tf.keras.layers.Conv3D(512, 3, padding="same")
        self.pool8   = tf.keras.layers.AveragePooling3D()

        self.flatten9 =  tf.keras.layers.Flatten()
        self.bn9      =  tf.keras.layers.BatchNormalization()
        self.dense9  = tf.keras.layers.Dense(2048, activation="relu", kernel_initializer='he_normal')
        self.drop9   =  tf.keras.layers.Dropout(self.drop_rate)
        
        self.bn10      =  tf.keras.layers.BatchNormalization()
        self.dense10  = tf.keras.layers.Dense(1024, activation="relu", kernel_initializer='he_normal')
        self.drop10   =  tf.keras.layers.Dropout(self.drop_rate)

        self.bn11        =tf.keras.layers.BatchNormalization()
        self.dense11  = tf.keras.layers.Dense(64, activation="softmax")

        if opt_name=="Adam":
            self.opt             = tf.keras.optimizers.Adam(self.alpha)
        if opt_name=="Sgd":
            self.opt             = tf.keras.optimizers.SGD(self.alpha)

    def call(self, inputs, training=None):
        outputs = inputs
        outputs = self.bn1(outputs, training=training)
        outputs = self.conv1(outputs)  
        outputs = self.pool1(outputs)

        outputs = self.bn2(outputs, training=training)        
        outputs = self.conv2(outputs)    
        outputs = self.pool2(outputs)

        outputs = self.bn3(outputs, training=training)
        outputs = self.conv3(outputs)

        outputs = self.bn4(outputs, training=training)
        outputs = self.conv4(outputs)
        outputs = self.pool4(outputs)

        outputs = self.bn5(outputs, training=training)
        outputs = self.conv5(outputs)

        outputs = self.bn6(outputs, training=training)
        outputs = self.conv6(outputs)
        outputs = self.pool6(outputs)

        outputs = self.bn7(outputs, training=training)
        outputs = self.conv7(outputs)

        outputs = self.bn8(outputs, training=training)
        outputs = self.conv8(outputs)
        outputs = self.pool8(outputs)

        outputs = self.flatten9(outputs)
        outputs = self.bn9(outputs, training=training)
        outputs = self.dense9(outputs)      
        outputs = self.drop9(outputs, training=training)

        outputs = self.bn11(outputs, training=training)
        outputs = self.dense11(outputs) 

        return outputs

    def train_step(self, videos, labels):
        #lossの計算方法は、https://danijar.com/variable-sequence-lengths-in-tensorflow/の"Masking the Cost Function"を参考にしました。
        with tf.GradientTape() as tape:
            pred = self(videos, True)
        #,weights=tf.compat.v1.to_float(mask),reduction=Reduction.None
            loss  =self.loss(labels, pred)

            loss_l2 = 0.0
            for v in self.trainable_variables:
                 loss_l2 = loss_l2 + self.lambd*tf.reduce_sum(v**2)/2
            loss = loss + loss_l2
            loss = tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size)
        grads   = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))

        self.train_acc.update_state(labels, pred)    
        return loss

    def test_step(self, videos, labels):
        pred = self(videos, False)
        loss  = self.loss(labels, pred)
        loss =  tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size)
        self.test_acc.update_state(labels, pred)        

        return loss

    @tf.function
    def distributed_train_step(self, videos, labels):
            return  self.mirrored_strategy.experimental_run_v2(self.train_step, args=(videos, labels))

    @tf.function
    def distributed_test_step(self, videos, labels):
            return  self.mirrored_strategy.experimental_run_v2(self.test_step,  args=(videos, labels))

    def train(self, train_ds, test_ds):
        with self.mirrored_strategy.scope():
            for epoch in range(self.epochs):

                train_mean_loss = 0.0
                test_mean_loss = 0.0
                num_batches = 0.0
                
                if (epoch+1)%10==0 :
                    self.opt.learning_rate = self.opt.learning_rate*0.1

                for (batch, (train_videos, train_labels)) in enumerate(train_ds): 

                    print("hparam : ", "epoch : ", epoch+1, "batch : ",batch+1)
                    losses            = self.distributed_train_step(train_videos, train_labels)
                    train_mean_loss       = train_mean_loss +self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,losses,axis=None)
                    num_batches += 1.0 
            
                train_mean_loss = train_mean_loss / num_batches
                print("the number of train batch : ", batch+1, "train_mean_loss : ", train_mean_loss, "train acc : ", self.train_acc.result())
                num_batches = 0.0
                for (batch, (test_videos, test_labels)) in enumerate(test_ds):
                    losses           = self.distributed_test_step(test_videos, test_labels)
                    test_mean_loss  =  test_mean_loss  + self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses,axis=None)
                    num_batches += 1.0
                filepath = os.getcwd()
                with self.mirrored_strategy.scope():
                    #self.set_weights(self.get_weights())
                
                    weights_name = filepath + "/weight/w"+"_"+str(self.alpha)+"_"+str(self.lambd)
                    self.save_weights(weights_name)
      
                self.y_list.append(self.train_acc.result())
                self.t_list.append(self.test_acc.result())
    
                test_mean_loss = test_mean_loss / num_batches
                print("the number of test batch : ", batch+1)
                print("epoch : ", epoch+1, " | train loss value : ", train_mean_loss.numpy(), ", test loss value : ", test_mean_loss.numpy())
                print("train acc : ", self.train_acc.result(), "test acc : ", self.test_acc.result())
                self.accuracy_reset()
                #weights_filename = "Model" + str(a) + str(l) + ".bin"
                #self.save_weights(weights_filename)#定期的に保存

    def accuracy_reset(self):
        self.train_acc.reset_states()
        self.test_acc.reset_states()

    def give_acc_list(self):
        return self.y_list, self.t_list
   