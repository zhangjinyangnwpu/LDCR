import tensorflow as tf
import numpy as np
import unit
import scipy.io as sio
import os
import matplotlib.pyplot as plt

class Model():

    def __init__(self,args,sess):
        self.args = args
        self.sess = sess
        info = sio.loadmat(os.path.join(self.args.tfrecords,'info.mat'))
        self.shape = info['shape']
        self.dim = info['dim']
        self.cube = args.cube
        self.tfrecords = args.tfrecords
        self.class_num = int(info['class_num'])
        self.data = info['data']
        self.imGIS = info['data_gt']
        self.total_train_num = info['total_train_num']
        self.result = args.result
        self.c_n = int(self.dim / args.c_r)
        self.data_path = args.data_path
        self.supervise_batch = args.supervise_batch # supervise train number
        self.ratio_cc = args.ratio_cc

        self.weight_learnable = args.weight_learnable
        self.global_step = tf.Variable(0,trainable=False)
        if args.use_lr_decay:
            self.lr = tf.train.exponential_decay(learning_rate=args.lr,
                                             global_step=self.global_step,
                                             decay_rate=args.decay_rete,
                                             decay_steps=args.decay_steps)
        else:
            self.lr = args.lr

        self.image = tf.placeholder(dtype=tf.float32, shape=(None, self.cube,self.cube,self.dim,1))
        self.label = tf.placeholder(dtype=tf.int64, shape=(None, 1))


        self.encoder = self.encoder
        self.decoder = self.decoder
        self.classifier = self.classifier

        self.feature = self.encoder(self.image)
        self.decode = self.decoder(self.feature)
        self.pre_label = self.classifier(self.feature)
        self.model_name = os.path.join('model.ckpt')
        self.loss()
        self.summary_write = tf.summary.FileWriter(os.path.join(self.args.log),graph=tf.get_default_graph())
        self.saver = tf.train.Saver(max_to_keep=100)

    def loss(self):
        with tf.name_scope('loss'):
            pp = int(self.cube//2)
            o_imge = self.image[:,pp:pp+1,pp:pp+1,:,:]
            self.o_imge = tf.layers.flatten(o_imge)
            print(self.o_imge,self.decode)
            self.loss_mse = tf.losses.mean_squared_error(self.o_imge,self.decode,scope='loss_mse')

            self.label_ = self.label[0:self.args.supervise_batch,:]
            self.pre_label_ = self.pre_label[0:self.args.supervise_batch,:]
            self.loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(self.label_,self.pre_label_,scope='loss_cross_entropy')
            self.loss_cross_entropy = tf.reduce_mean(self.loss_cross_entropy)

            self.alpha = tf.Variable(initial_value=tf.constant(self.args.ratio_cc,dtype=tf.float32))# for mse
            self.beta = tf.Variable(initial_value=tf.constant(1,dtype=tf.float32))# for crossentropy

            if self.weight_learnable:
                self.loss_total = self.loss_mse*self.alpha + self.loss_cross_entropy*self.beta
            else:
                self.loss_total = self.loss_mse * self.args.ratio_cc + self.loss_cross_entropy

            tf.add_to_collection('losses',self.loss_total)

            tf.summary.scalar('loss_cross_entropy',self.loss_cross_entropy)
            tf.summary.scalar('loss_mse',self.loss_mse)
            tf.summary.scalar('loss_total',self.loss_total)
            tf.summary.scalar('learning_rate',self.lr)

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total,global_step=self.global_step)
        self.merged = tf.summary.merge_all()


    def encoder(self,image):
        print(image)
        f_num = 512
        k_init = tf.initializers.variance_scaling
        act_f = tf.nn.relu
        with tf.variable_scope('encoder'):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv3d(image,f_num,(1,1,8),(1,1,3),padding='valid',activation=act_f,
                                         kernel_initializer=k_init)
                conv0 = tf.layers.batch_normalization(conv0)
                # conv0 = tf.nn.relu(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(conv0,f_num,(self.cube,self.cube,3),(1,1,2),padding='valid',activation=act_f,
                                         kernel_initializer=k_init)
                conv1 = tf.layers.batch_normalization(conv1)
                # conv1 = tf.nn.relu(conv1)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(conv1,f_num,(1,1,3),(1,1,2),padding='valid',activation=act_f,
                                         kernel_initializer=k_init)
                conv2 = tf.layers.batch_normalization(conv2)
                # conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.variable_scope('conv3'):
                self.f_shape = int(conv2.get_shape().as_list()[3])
                conv3 = tf.layers.conv3d(conv2,f_num,(1,1,self.f_shape),(1,1,1),padding='valid',activation=act_f,
                                         kernel_initializer=k_init)
                conv3 = tf.layers.batch_normalization(conv3)
                # conv3 = tf.nn.relu(conv3)
                print(conv3)
            with tf.variable_scope('conv4'):
                feature = tf.layers.conv3d(conv3,self.c_n,(1,1,1),(1,1,1),padding='valid')
                print(feature)
            feature = tf.layers.flatten(feature)
            print(feature)
        return feature


    def decoder(self,image):
        image = tf.expand_dims(image, 1)
        image = tf.expand_dims(image, 2)
        print(image)
        f_num = 512
        k_init = tf.initializers.variance_scaling
        act_f = tf.nn.relu
        with tf.variable_scope('decoder'):
            with tf.variable_scope('de_1_1'):
                image = tf.layers.conv2d(image,f_num,(1,1),(1,1),activation=act_f,
                                         kernel_initializer=k_init)
            with tf.variable_scope('de_conv0'):
                conv0 = tf.layers.conv2d_transpose(image,f_num,(self.f_shape,1),(1,1),activation=act_f,
                                         kernel_initializer=k_init)
                conv0 = tf.layers.batch_normalization(conv0)
                # conv0 = tf.nn.relu(conv0)
                print(conv0)
            with tf.variable_scope('de_conv1'):
                conv1 = tf.layers.conv2d_transpose(conv0, f_num, (3, 1), (2, 1), activation=act_f,
                                                   kernel_initializer=k_init)
                conv1 = tf.layers.batch_normalization(conv1)
                # conv1 = tf.nn.relu(conv1)
                print(conv1)
            with tf.variable_scope('de_conv2'):
                conv2 = tf.layers.conv2d_transpose(conv1, f_num, (4, 1), (2, 1), activation=act_f,
                                                   kernel_initializer=k_init)
                conv2 = tf.layers.batch_normalization(conv2)
                # conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.variable_scope('de_conv3'):
                conv3 = tf.layers.conv2d_transpose(conv2, f_num, (10, 1), (3, 1), activation=act_f,
                                                   kernel_initializer=k_init)
                conv3 = tf.layers.batch_normalization(conv3)
                # conv3 = tf.nn.relu(conv3)
                print(conv3)
            with tf.variable_scope('de_image'):
                d_im = tf.layers.conv2d(conv3,1,(1,1),(1,1))
                print(d_im)
            de_image = d_im
            de_image = tf.layers.flatten(de_image)
            print(de_image)
        return de_image


    def classifier(self, feature):
        feature = tf.expand_dims(feature, 2)
        f_num = 16
        print(feature)
        with tf.variable_scope('classifer', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv1d(feature, f_num, (8), strides=(1), padding='same')
                conv0 = tf.layers.batch_normalization(conv0)
                conv0 = tf.nn.relu(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv1d(conv0, f_num * 2, (3), strides=(1), padding='same')
                conv1 = tf.layers.batch_normalization(conv1)
                conv1 = tf.nn.relu(conv1)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv1d(conv1, f_num * 4, (3), strides=(1), padding='same')
                conv2 = tf.layers.batch_normalization(conv2)
                conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.variable_scope('global_info'):
                f_shape = conv2.get_shape()
                feature = tf.layers.conv1d(conv2, self.class_num, (int(f_shape[1])), (1))
                feature = tf.layers.flatten(feature)
                print(feature)
        return feature

    def load(self, checkpoint_dir):
        print("Loading model ...")
        model_name = os.path.join(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(model_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_name, ckpt_name))
            print("Load successful.")
            return True
        else:
            print("Load fail!!!")
            exit(0)

    def train(self,traindata,saedata,data_model):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.iterate_num = self.args.iter_num
        oa_list,aa_list,kappa_list,ac_llist,matrix_list = list(),list(),list(),list(),list()
        psnr_list = list()
        loss_list,mse_list,ce_list = list(),list(),list()
        for i in range(self.iterate_num):
            (train_data,train_label),(row,col,sae_data) = self.sess.run([traindata,saedata])
            # print(train_data.shape,train_label.shape,sae_data.shape)
            if train_label.shape[0] != self.supervise_batch:
                continue
            feed_data = np.concatenate([train_data,sae_data])
            _,summery,lr= self.sess.run([self.optimizer,self.merged,self.lr],
                                                feed_dict={self.image:feed_data,self.label_:train_label})

            if i % 1000 == 0:
                l_ce, l_mse, l_t = self.sess.run(
                    [self.loss_cross_entropy, self.loss_mse, self.loss_total],
                    feed_dict={self.image: feed_data, self.label_: train_label})
                print('step:%d crossentropy:%f mes:%f total:%f lr:%f '%(i,l_ce, l_mse*self.ratio_cc, l_t,lr))
                loss_list.append(l_t)
                mse_list.append(l_mse)
                ce_list.append(l_ce)
                sio.savemat(os.path.join(self.result,'loss_list.mat'),{'loss_t':loss_list,
                                                                    'mse':mse_list,
                                                                    'ce':ce_list})
            if i % 10000 == 0:
                self.saver.save(self.sess, os.path.join(self.args.model, self.model_name), global_step=i)
                dataset_test = data_model.data_parse(
                    os.path.join(self.tfrecords, 'test_data.tfrecords'), type='test')
                dataset_sae_test = data_model.data_parse(
                    os.path.join(self.tfrecords, 'sae_test_data.tfrecords'), type='sae_test')
                oa, aa, kappa, ac_list, matrix = self.test(dataset_test)
                oa_list.append(oa)
                aa_list.append(aa)
                kappa_list.append(kappa)
                ac_llist.append(ac_list)
                matrix_list.append(matrix)
                sio.savemat(os.path.join(self.result, 'result_list.mat'),
                            {'oa': oa_list, 'aa': aa_list, 'kappa': kappa_list, 'ac_list': ac_llist,
                             'matrix': matrix_list})
                psnr = self.get_decode_image(dataset_sae_test)
                psnr_list.append(psnr)
                sio.savemat(os.path.join(self.result, 'psnr_list.mat'), { 'psnr': psnr_list})
            self.summary_write.add_summary(summery,i)

    def test(self,testdata):
        acc_num,test_num = 0,0
        matrix = np.zeros((self.class_num,self.class_num),dtype=np.int32)
        try:
            while True:
                test_data, test_label = self.sess.run(testdata)
                # print(test_data.shape,test_label.shape)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:test_data,self.label:test_label})
                pre_label = np.argmax(pre_label,1)
                pre_label = np.expand_dims(pre_label,1)
                acc_num += np.sum((pre_label==test_label))
                test_num += test_label.shape[0]
                print(acc_num,test_num,acc_num/test_num)
                for i in range(pre_label.shape[0]):
                    matrix[pre_label[i],test_label[i]]+=1
        except tf.errors.OutOfRangeError:
            print("test end!")

        ac_list = []
        for i in range(len(matrix)):
            ac = matrix[i, i] / sum(matrix[:, i])
            ac_list.append(ac)
            print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
        print('confusion matrix:')
        print(np.int_(matrix))
        print('total right num:', np.sum(np.trace(matrix)))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print('oa:', accuracy)
        # kappa
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        ac_list = np.asarray(ac_list)
        aa = np.mean(ac_list)
        oa = accuracy
        print('aa:',aa)
        print('kappa:', kappa)

        sio.savemat(os.path.join(self.result, 'result.mat'), {'oa': oa,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})
        return oa,aa,kappa,ac_list,matrix

    def save_decode_map(self,sae_data):
        data_gt = self.imGIS
        # plt.figure(figsize=(map.shape[1] / 5, map.shape[0] / 5), dpi=100)# set size
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        plt.axis('off')
        plt.pcolor(data_gt, cmap='jet')
        plt.savefig(os.path.join(self.result, 'groundtrouth.png'), format='png')
        plt.close()
        print('Groundtruth map get finished')
        de_map = np.zeros(data_gt.shape,dtype=np.int32)
        try:
            while True:
                row,col,map_data = self.sess.run(sae_data)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:map_data})
                pre_label = np.argmax(pre_label,1)
                for i in range(pre_label.shape[0]):
                    c,l = row[i],col[i]
                    de_map[c,l] = pre_label[i]+1
        except tf.errors.OutOfRangeError:
            print("test end!")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        plt.axis('off')
        plt.pcolor(de_map, cmap='jet')
        plt.savefig(os.path.join(self.result, 'decode_map.png'), format='png')
        plt.close()
        print('decode map get finished')

    def get_decode_image(self,data):
        de_image = np.zeros(shape=self.shape[0])
        try:
            while True:
                row,col,feed_data = self.sess.run(data)
                de_pixel = self.sess.run(self.decode, feed_dict={self.image: feed_data})
                for k in range(de_pixel.shape[0]):
                    de_image[row[k],col[k],:] = de_pixel[k]
        except tf.errors.OutOfRangeError:
            print("get decode image end!")
        psnr = unit.PSNR(self.data,de_image)
        print('reconstructed PSNR:',psnr)
        sio.savemat(os.path.join(self.result,'decode_image.mat'),{'decode_image':de_image,'psnr':psnr})
        return psnr

    def get_feature(self,data):
        # print(self.shape[0])
        shape = self.shape[0]
        shape[2] = self.c_n
        de_image = np.zeros(shape=shape)
        try:
            while True:
                row,col,feed_data = self.sess.run(data)
                de_pixel = self.sess.run(self.feature, feed_dict={self.image: feed_data})
                for k in range(de_pixel.shape[0]):
                    de_image[row[k],col[k],:] = de_pixel[k]
        except tf.errors.OutOfRangeError:
            print("get decode image end!")
        # psnr = unit.PSNR(self.data,de_image)
        # print('reconstructed PSNR:',psnr)
        sio.savemat(os.path.join(self.result,'feature.mat'),{'feature':de_image})
        # return psnr
