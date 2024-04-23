import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
from tensorflow.keras.utils import save_img
import matplotlib.pyplot as plt
import datetime
import time
import csv
import os

from .dataset import dataset_from_multiple_directrory
from .plor_func import plot_multiple_images, plot_loss_log
from CSPPGAN_model.Encoder_model import make_encoder
from CSPPGAN_model.Generator_model import make_generator
from CSPPGAN_model.Discriminator_model import make_multi_scale_discriminator

class CSPPGAN_Trainer():
    def __init__(self, savepath="/CSPPGAN_save"):
        self.savepath = savepath
        self.input_shape = (128, 128, 3)
        self.n_batchs = 0
        self.n_trainset = 0
        self.run_epochs = 0
        self.save_epoch = 100
        self.show_epoch = 10
        self.num_train_step_return = 20
        self.today = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        self.group_train_dataset = []
        self.group_test_dataset = []
        self.log_list_tmp = []
        self.log_list = [[] for i in range(self.num_train_step_return)]

    def setup_group_dataset(self, path_sketch_list, dataset_type, size=None, num=None, shuffle=True):
        path_list = []
        sketch_list = []
        for tmp in path_sketch_list:
            path_list.append(tmp[0])
            sketch_list.append(tmp[1])
        
        if dataset_type == "train":
            self.n_trainset = self.n_trainset or num
            size = size or ((self.batch_size, ) + self.input_shape)
            self.n_batchs, self.group_train_dataset = dataset_from_multiple_directrory(path_list, size=size, num=num, 
                                                                                       shuffle=shuffle, sketch_list=sketch_list)
            
        elif dataset_type == "test":
            num = num or self.n_valset
            size = size or ((self.n_valset, ) + self.input_shape)
            _, self.group_test_dataset = dataset_from_multiple_directrory(path_list, size=size, num=num,
                                                                          shuffle=shuffle, sketch_list=sketch_list)
        
    def loss_record(self, result):
        color = 'rgbykcm'
        
        self.log_list_tmp.append([float(x) / self.n_batchs for x in result])
        for i, x in enumerate(result):
            self.log_list[i].append(x / self.n_batchs)
            plot_loss_log(self.log_list[i], color[i % len(color)])
  
    def save_model(self, model, model_dir, model_name):
        savepath = self.savepath + "/" + model_dir

        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        modelpath = savepath + "/" + model_name + "_{}epochs_{}set.h5". format(self.run_epochs, self.n_trainset)
        model.save(modelpath)
            
    def save_loss(self):
        savepath = self.savepath
        csv_name = self.csv_name
        
        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        csvpath = savepath + "/" + csv_name + "_log_{}.csv". format(self.today)
        with open(csvpath, 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.log_list_tmp)
        self.log_list_tmp = []
        
    def save(self):
        self.save_loss()

        if self.trainer_type == "encoder":
            self.save_model(self.encoder, model_dir="encoder_h5", model_name="CSPPGAN_en")

        elif self.trainer_type == "gan":
            self.save_model(self.generator, model_dir=self.gen_dir, model_name=self.dis_name)
            self.save_model(self.discriminator, model_dir=self.dis_dir, model_name=self.dis_name)        

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            print("Epoch {}/{}".format(epoch + 1, n_epochs))
            start_time = time.time()
            
            batch_result = [0 for i in range(self.num_train_step_return)]
            for i, data in enumerate(zip(*self.group_train_dataset)):
                result = self.train_step(data)
                batch_result = [x + y for x, y in zip(batch_result, result)]
                print("{} / {}".format(i, self.n_batchs), end='\r')
            print("{} / {}".format(self.n_batchs, self.n_batchs))
            
            self.run_epochs += 1
            if self.run_epochs % self.save_epoch == 0:
                self.show_result()
                self.loss_record(batch_result)
                self.save()
            print('Time for epoch {} is {} sec' .format(epoch+1, int(time.time()-start_time)))

            
class CSPPGAN_Encoder_Trainer(CSPPGAN_Trainer):
    def __init__(self, csv_name="CSPPGAN_en"):
        super().__init__()
        self.trainer_type = "encoder"
        self.csv_name = csv_name
        self.batch_size = 50
        self.n_valset = 10
        self.encoder = None
        self.optimizer = None
        self.mae = tf.keras.losses.MeanAbsoluteError()

    def make_model(self, lr=0.00005):
        self.encoder = make_encoder(self.input_shape)
        self.optimizer = optimizers.RMSprop(lr)
                   
    def loss_function(self, x, x_p, y, y_p):
        loss = self.mae(x, x_p) + self.mae(y, y_p)
        return loss
    
    def show_result(self):
        for x, y in zip(*self.group_test_dataset):
            for imgs in [x, self.encoder(x)[0], y, self.encoder(y)[0]]:
                plot_multiple_images(imgs)
                plt.show()
                            
    @tf.function
    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            x, y = data[0], data[1]
            x_result = self.encoder(x, training=True)
            y_result = self.encoder(y, training=True)
           
            loss = self.loss_function(x, x_result[0], y, y_result[0])

        grad = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.encoder.trainable_variables))

        return [loss]


class CSPPGAN_GAN_Trainer(CSPPGAN_Trainer):
    def __init__(self, gen_dir="gen_h5", dis_dir="dis_h5", gen_name="CSPPGAN_gen", dis_name="CSPPGAN_dis", csv_name="CSPPGAN_gan"):
        super().__init__()
        self.trainer_type = "gan"
        self.csv_name = csv_name
        self.gen_dir = gen_dir
        self.dis_dir = dis_dir
        self.gen_name = gen_name
        self.dis_name = dis_name
        self.batch_size = 8
        self.n_valset = 10
        self.encoder = None
        self.generator = None
        self.discriminator = None
        self.gen_optimizer = None
        self.dis_optimizer = None
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.dfm = 1
        self.boost = 1
        self.rec = 1

    def make_model(self, encoder_path, gen_lr=0.00008, dis_lr=0.001):
        self.encoder = models.load_model(encoder_path)
        
        self.generator = make_generator()
        self.discriminator = make_multi_scale_discriminator(self.input_shape)
        self.gen_optimizer = optimizers.RMSprop(gen_lr)
        self.dis_optimizer = optimizers.RMSprop(dis_lr)
                   
    def loss_function(self, z, g, d_x, d_z, dfm=1, boost=1, rec=1):
        def adv_loss(output):
            loss = self.ce(tf.ones_like(output), output)
            return loss
        
        dfm_loss = self.mae(d_x[3], d_z[3]) + self.mae(d_x[4], d_z[4]) + self.mae(d_x[5], d_z[5]) \
                 + self.mae(d_x[6], d_z[6]) + self.mae(d_x[7], d_z[7]) + self.mae(d_x[8], d_z[8])
        boost_loss = self.mae(g[1], g[2])
        rec_loss = self.mae(g[3], z) + self.mae(g[4], z) + self.mae(g[5], z)

        g_adv_loss = adv_loss(d_x[0] - d_z[0]) + adv_loss(d_x[1] - d_z[1]) + adv_loss(d_x[2] - d_z[2])
        d_adv_loss = adv_loss(d_z[0] - d_x[0]) + adv_loss(d_z[1] - d_x[1]) + adv_loss(d_z[2] - d_x[2])

        loss_gen = g_adv_loss + dfm * dfm_loss + boost * boost_loss + rec * rec_loss
        loss_dis = d_adv_loss

        return [loss_gen, loss_dis, dfm_loss, boost_loss, rec_loss]
    
    def show_result(self):
        for x, y, z in zip(*self.group_test_dataset):
            x_result = self.encoder(x)
            y_result = self.encoder(y)
            z_result = self.encoder(z)

            for imgs in [x, z] + self.generator([x_result[1], y_result[1], z_result[2]]):
                plot_multiple_images(imgs)
                plt.show()
            
    def save_img(self, savepath=None):
        savepath = savepath or self.savepath + "/" + "result"
        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        for p in ["X", "Y", "Z"]:
            if not os.path.isdir(savepath + "/" + p):
                os.makedirs(savepath + "/" + p)

        count = 0
        for x, y, z in zip(*self.group_train_dataset):
            x_result = self.encoder(x)
            y_result = self.encoder(y)
            z_result = self.encoder(z)

            for i, imgs in enumerate(zip(x, z)):
                save_img(savepath + "/X/{:03}.jpg".format(count*self.batch_size + i), imgs[0])
                save_img(savepath + "/Z/{:03}.jpg".format(count*self.batch_size + i), imgs[1])
            
            for p, imgs in zip(["X0", "X1", "X2", "Y0", "Y1", "Y2"], self.generator([x_result[1], y_result[1], z_result[2]])):
                path = savepath + "/Y/" + p
                if not os.path.isdir(path):
                    os.makedirs(path)
                    
                for i, img in enumerate(imgs):    
                    save_img(path + "/{:03}.jpg".format(count*self.batch_size + i), img)
            count += 1
        
    @tf.function
    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            x, y, z = data[0], data[1], data[2]
            
            x_result = self.encoder(x, training=False)
            y_result = self.encoder(y, training=False)
            z_result = self.encoder(z, training=False)

            gen = self.generator([x_result[1], y_result[1], z_result[2]], training=True)
            dis_x = self.discriminator(gen[2], training=True)
            dis_z = self.discriminator(z, training=True)
           
            loss = self.loss_function(z, gen, dis_x, dis_z, dfm=self.dfm, boost=self.boost, rec=self.rec)

        grad_gen = tape.gradient(loss[0], self.generator.trainable_variables)
        grad_dis = tape.gradient(loss[1], self.discriminator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        self.dis_optimizer.apply_gradients(zip(grad_dis, self.discriminator.trainable_variables))

        return loss
