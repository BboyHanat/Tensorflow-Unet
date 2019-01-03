import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import cv2
import tensorflow.contrib.slim as slim
import resnet_v1,resnet_utils
import os
resnet_arg_scope = resnet_utils.resnet_arg_scope



class U_Net():
    _networks_map = {
        'resnet_v1_50': {'C1': 'resnet_v1_50/conv1',
                     'C2': 'resnet_v1_50/block1/unit_2/bottleneck_v1',
                     'C3': 'resnet_v1_50/block2/unit_3/bottleneck_v1',
                     'C4': 'resnet_v1_50/block3/unit_5/bottleneck_v1',
                     'C5': 'resnet_v1_50/block4/unit_3/bottleneck_v1',
                     },
        'resnet_v1_101': {'C1': '', 'C2': '',
                      'C3': '', 'C4': '',
                      'C5': '',
                      }}

    def __init__(self, batch_size, width, height,
                 channels, pretrained_model, class_num, backbones="resnet_v1_50"):

        self.images = tf.placeholder(tf.float32,shape=[None,None,None,channels])
        self.labels = tf.placeholder(tf.int64,shape=[None,None,None,1])
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.channels = channels
        self.backbones = backbones
        self.class_num = class_num
        self.pretrained_model = pretrained_model

    def create_graph(self, sess, global_step, training_iters,learning_rate=0.001, decay_rate=0.95, momentum=0.2,):
        with sess.graph.as_default():
            pyramid_map = self._networks_map[self.backbones]
            net, end_points = resnet_v1.resnet_v1_50(self.images)
            p5 = end_points[pyramid_map["C5"]]
            p4 = end_points[pyramid_map["C4"]]
            p3 = end_points[pyramid_map["C3"]]
            p2 = end_points[pyramid_map["C2"]]
            p1 = end_points[pyramid_map["C1"]]
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'updates_collections': tf.GraphKeys.UPDATE_OPS,
                'fused': None,  # Use fused batch norm if possible.
            }
            with tf.variable_scope("ResNet_Unet", [p5,p4,p3,p2,p1]) as sc:
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(0.0001),
                                    weights_initializer=slim.variance_scaling_initializer(),
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm if True else None,
                                    normalizer_params=batch_norm_params):
                    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                        "attention"
                        p5_64 = slim.conv2d(p5, 64, [3, 3], stride=1, scope='channel_64_p5_conv3')      ##channel form 2048 to 64
                        p5_64 = slim.conv2d(p5_64, 64, [1, 1], stride=1, activation_fn=None, scope='channel_64_p5_conv1')
                        "channel se"
                        p5_cse =tf.reduce_mean(p5, [1,2], keep_dims=True)
                        p5_cse_conv = slim.conv2d(p5_cse, 64, [1, 1], stride=1, activation_fn=None, scope='cse_p5_conv')
                        p5_cse_sig = tf.sigmoid(p5_cse_conv, name='cse_p5_sig')
                        p5_cse = p5_64*p5_cse_sig
                        "spatial se"
                        p5_sse = slim.conv2d(p5, 1, [1, 1], stride=1, activation_fn=None, scope='sse_p5_conv')
                        p5_sse_sig = tf.sigmoid(p5_sse, name='cse_p5_sig')
                        p5_sse = p5_64 * p5_sse_sig
                        "shortcut"
                        p5_64_shortcut = slim.conv2d(p5, 64, [1, 1], stride=1, activation_fn=None,scope='channel_64_p5_shortcut')
                        p5_att=p5_cse+p5_sse+p5_64_shortcut



                        '''gau'''
                        p5_gp=tf.reduce_mean(p5_att,[1,2],keep_dims=True)
                        p5_gp_conv = slim.conv2d(p5_gp,64,[1,1],stride=1, activation_fn=None,scope='gau_conv_p5gp')
                        p5_gp_conv_sig= tf.sigmoid(p5_gp_conv, name='gau_p5_sig')
                        p4_conv=slim.conv2d(p4, 64, [3,3], stride=1, scope='gau_conv_p4')
                        p4_conv = slim.conv2d(p4_conv, 64, [1, 1], stride=1, activation_fn=None, scope='p4_conv1')
                        p4_attention=p4_conv*p5_gp_conv_sig
                        p5_up=tf.image.resize_bilinear(p5_att, [tf.shape(p4)[1], tf.shape(p4)[2]], name='gau_p5_upscale')
                        p4_add=p4_attention+p5_up
                        '''gau'''
                        p4_gp = tf.reduce_mean(p4_add, [1, 2],keep_dims=True)
                        p4_gp_conv = slim.conv2d(p4_gp, 64, [1, 1], stride=1, activation_fn=None,scope='gau_conv_p4gp')
                        p4_gp_conv_sig = tf.sigmoid(p4_gp_conv, name='gau_p4_sig')
                        p3_conv = slim.conv2d(p3, 64, [3, 3], stride=1, scope='gau_conv_p3')
                        p3_conv = slim.conv2d(p3_conv, 64, [1, 1], stride=1, activation_fn=None, scope='p3_conv1')
                        p3_attention = p3_conv * p4_gp_conv_sig
                        p4_up = tf.image.resize_bilinear(p4_add, [tf.shape(p3)[1], tf.shape(p3)[2]],
                                                         name='gau_p4_upscale')
                        p3_add = p3_attention + p4_up
                        '''gau'''
                        p3_gp = tf.reduce_mean(p3_add, [1, 2],keep_dims=True)
                        p3_gp_conv = slim.conv2d(p3_gp, 64, [1, 1], stride=1, activation_fn=None,scope='gau_conv_p3gp')
                        p3_gp_conv_sig = tf.sigmoid(p3_gp_conv, name='gau_p3_sig')
                        p2_conv = slim.conv2d(p2, 64, [3, 3], stride=1,scope='gau_conv_p2')
                        p2_conv = slim.conv2d(p2_conv, 64, [1, 1], stride=1, activation_fn=None, scope='p2_conv1')
                        p2_attention = p2_conv * p3_gp_conv_sig
                        p3_up = tf.image.resize_bilinear(p3_add, [tf.shape(p2)[1], tf.shape(p2)[2]],
                                                         name='gau_p3_upscale')
                        p2_add = p2_attention + p3_up

                        '''gau'''
                        p2_gp = tf.reduce_mean(p2_add, [1, 2], keep_dims=True)
                        p2_gp_conv = slim.conv2d(p2_gp, 64, [1, 1], stride=1, activation_fn=None, scope='gau_conv_p2gp')
                        p2_gp_conv_sig = tf.sigmoid(p2_gp_conv, name='gau_p2_sig')
                        p1_conv = slim.conv2d(p1, 64, [3, 3], stride=1, scope='gau_conv_p1')
                        p1_conv = slim.conv2d(p1_conv, 64, [1, 1], stride=1, activation_fn=None, scope='p1_conv1')
                        p1_attention = p1_conv * p2_gp_conv_sig
                        p2_up = tf.image.resize_bilinear(p2_add, [tf.shape(p1)[1], tf.shape(p1)[2]],
                                                         name='gau_p2_upscale')
                        p1_add = p1_attention + p2_up
                        print("p1 is ", p1_add.get_shape())

                        outputs_128 = tf.image.resize_bilinear(p1_add, [tf.shape(self.images)[1], tf.shape(self.images)[2]], name='gau_p2_upscale')

                        outputs_128 = slim.conv2d(outputs_128, 64, [3, 3], stride=1, scope="output_mask1")
                        outputs2 = slim.conv2d(outputs_128, self.class_num, [3, 3], stride=1, scope="output_mask2", activation_fn=None)
                        self.outputs3 = slim.conv2d(outputs2, self.class_num, [3, 3], stride=1, scope="output_mask3", activation_fn=None)

            self.output_softmax = self.pixel_wise_softmax(self.outputs3)
            self.mask = tf.argmax(self.output_softmax,axis=3,name="Mask")
            oneHot_mask = tf.one_hot(self.mask, self.class_num)
            self.oneHot_mask_flatten = slim.flatten(oneHot_mask)
            oneHot_label = tf.one_hot(self.labels,self.class_num)
            self.labels_flatten=slim.flatten(oneHot_label)
            self.oneHot_label_reshape=tf.reshape(oneHot_label,[-1,self.class_num])
            self.output_softmax_reshape = tf.reshape(self.output_softmax, [-1, self.class_num])

            self.d_loss = self.dice_coefficient_loss(self.labels_flatten,self.oneHot_mask_flatten)
            #self.loss = -tf.reduce_mean(self.labels_flatten * tf.log(tf.clip_by_value(self.outputs_flatten, 1e-10, 1.0)))
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.oneHot_label_reshape,logits=self.output_softmax_reshape))
            self.t_loss = self.loss + self.d_loss

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters*4,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
            #                                        ).minimize(self.loss, global_step=global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.t_loss,
                                                                           global_step=global_step)

    def dice_coefficient_loss(self,labels,outputs):
        eps = 1e-5
        self.intersection = tf.reduce_sum(labels*outputs)
        self.union = tf.reduce_sum(labels)+tf.reduce_sum(outputs)+eps
        return -tf.log(2*self.intersection / (self.union))


    def pixel_wise_softmax(self, output_map):
        with tf.name_scope("pixel_wise_softmax"):
            self.max_axis = tf.reduce_max(output_map, axis=3, keep_dims=True)
            self.exponential_map = tf.exp(output_map - self.max_axis)
            normalize = tf.reduce_sum(self.exponential_map, axis=3, keep_dims=True)
            return self.exponential_map / normalize


    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []
        for v in variables:
            # exclude
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore
    #load pretrained model from the path
    def load_pretrained_model(self,sess):
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))
        print("variables initilized ok")
        #Get dictionary of model variable
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # # Get the variables to restore
        variables_to_restore = self.get_variables_to_restore(variables, var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)


    def train_unet(self, sess, image, label, epochs, train_image_num,learning_rate=0.001,show_step=50):
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step_unet")
        training_iters = train_image_num
        self.create_graph(sess, global_step, training_iters,learning_rate=learning_rate)
        print("create graph ok")
        self.load_pretrained_model(sess)
        print("load pretrained model ok")
        with sess.as_default():
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # for epoch in range(epochs):
            #     for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
            #         batch_x, batch_y = sess.run([image, label])
            #         #batch_x = sess.run(image)
            #         print(step)
            #         print(step)
            # coord.request_stop()
            # coord.join(threads)
            print("Start Training")
            saver = tf.train.Saver(max_to_keep=4)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for epoch in range(epochs):
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    batch_x,batch_y = sess.run([image,label])
                    batch_x=batch_x-114.8
                    #batch_y[batch_y>0]=1
                    #batch_y=np.clip(batch_y,0,1)
                    # intersection, union = sess.run([self.intersection, self.union],
                    #                               feed_dict={self.images: batch_x, self.labels: batch_y})
                    # print(intersection)
                    # print(union)
                    #print(batch_y)

                    # batch_x=np.reshape(batch_x,(self.batch_size, self.height, self.width, self.channels))
                    # batch_y = np.reshape(batch_y[:,:,:,0:1], (self.batch_size, self.height, self.width, 1))
                    mask, loss, d_loss = sess.run([self.mask, self.loss, self.d_loss],
                                                      feed_dict={self.images: batch_x, self.labels: batch_y})
                    #print(" loss is: ", loss, " d_loss is: ", d_loss, "epoch is: ", epoch, "step is: ", step)
                    # print("Test")
                    sess.run(self.optimizer,feed_dict={self.images: batch_x, self.labels: batch_y})
                    if step % show_step == 0 and step >= show_step:
                        print(" loss is: ", loss, " d_loss is: ", d_loss, "epoch is: ", epoch, "step is: ",step)
                        mask=np.reshape(mask[0,:,:],(self.height,self.width,1))

                        mask = mask*255
                        image_to_show = np.clip(mask,0,255)
                        image_to_show = np.asarray(image_to_show, np.uint8)
                        save_path=os.path.join("/data/Hanati_Presonel/Unet/results",str(epoch)+"_"+str(step)+".jpg")
                        cv2.imwrite(save_path,image_to_show)

                ckpt_name = 'Resnet50_Unet' + str(epoch) + '.ckpt'
                saver.save(sess, "./ckpt/" + ckpt_name)
            coord.request_stop()
            coord.join(threads)

    def test(self,sess,test_image,test_label):
        image,label=sess.run([test_image,test_label])

    def forward_unet(self,sess,image,show=False):
        image = cv2.resize(image, (self.height, self.width))
        image=np.asarray(image,np.float32)
        image=np.reshape(image,(1,self.height,self.width,self.channels))-114.8
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        self.output_sigmoid = tf.get_default_graph().get_tensor_by_name("Sigmoid:0")
        output_sigmoid=sess.run([self.output_sigmoid],feed_dict={self.images_placeholder: image})
        output_sigmoid = np.reshape(output_sigmoid, (-1, self.height,self.width, 1))
        image_to_show = output_sigmoid[0, :, :, :] * 255
        image_to_show = np.asarray(image_to_show, np.uint8)

        if show:
            cv2.imshow("mask", image_to_show)
            cv2.waitKey(0)
        return image_to_show

