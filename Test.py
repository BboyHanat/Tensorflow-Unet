from model import U_Net
import numpy as np
import tensorflow as tf
import os,cv2
import glob
os.environ['CUDA_VISIBLE_DEVICES']='0'
def forward(image_path,save_path):
    height = 129
    width = 129
    channels = 3
    batch_size = 1
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("ckpt/Resnet50_Unet5.ckpt.meta", input_map=None)
        saver.restore(tf.get_default_session(), "ckpt/Resnet50_Unet5.ckpt")
    path_frames = [os.path.join(image_path, file) for file in os.listdir(image_path)
                   if file.endswith(".png")]
    unet = U_Net(batch_size, width, height, channels, pretrained_model='resnet_v1_50.ckpt')
    for path in path_frames:
        img = cv2.imread(path)
        img_name=path.split("/")[-1]
        img = cv2.resize(img, (101, 101))
        mask=unet.forward_unet(sess,img)
        cv2.imwrite(os.path.join(save_path,img_name),mask)


def train():
    train_img_root = "/data/GoogleDownloads/food_imgs/IMG/"
    train_label_root = "/data/GoogleDownloads/food_imgs/GT/"
    test_img_root = "./dataset/images_prepped_test/"
    test_label_root = "./dataset/annotations_prepped_test/"
    height = 224
    width = 224
    channels = 3
    class_num=67
    batch_size = 16
    epoch = 64
    show_step = 50

    #train_images_path = glob.glob(train_img_root + "*.jpg") + glob.glob(train_img_root + "*.png") + glob.glob(train_img_root + "*.jpeg")
    train_images_path=[os.path.join(train_img_root, img_path) for img_path in os.listdir(train_img_root)
                       if os.path.isfile(os.path.join(train_img_root, img_path))]
    print(train_images_path)
    train_labels_path = [os.path.join(train_label_root, img_path.split("/")[-1]) for img_path in train_images_path if
                     os.path.isfile(os.path.join(train_label_root, img_path.split("/")[-1]))]
    print(train_labels_path)
    test_images_path = glob.glob(test_img_root + "*.jpg") + glob.glob(test_img_root + "*.png") + glob.glob(test_img_root + "*.jpeg")
    test_labels_path = [os.path.join(test_label_root, img_path.split("/")[-1]) for img_path in test_images_path if
                     os.path.isfile(os.path.join(test_label_root, img_path.split("/")[-1]))]
    print(len(train_labels_path),len(train_images_path))
    assert len(train_labels_path) == len(train_images_path)
    assert len(test_images_path) == len(test_labels_path)
    for im, seg in zip(train_images_path, train_labels_path):
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
    for im, seg in zip(test_images_path, test_labels_path):
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    train_image_num = len(train_images_path)
    print(train_labels_path)
    sess=tf.Session()

    '''train_data'''
    train_img_queue = tf.train.string_input_producer(train_images_path, shuffle=False)
    train_label_queue = tf.train.string_input_producer(train_labels_path, shuffle=False)
    image_reader1 = tf.WholeFileReader()
    image_reader2 = tf.WholeFileReader()
    _, train_img_file = image_reader1.read(train_img_queue)
    _, train_label_file = image_reader2.read(train_label_queue)
    train_img=tf.image.resize_images(tf.image.decode_jpeg(train_img_file), [height, width])
    train_label_file=tf.image.decode_jpeg(train_label_file)
    train_label_file=tf.expand_dims(train_label_file, 0)
    train_label = tf.image.resize_nearest_neighbor(train_label_file, [height, width])
    train_img=tf.cast(tf.reshape(train_img,[height,width,3]),tf.float32)
    train_label = tf.cast(tf.reshape(train_label, [height, width, 1]), tf.float32)
    img_train_batch,label_train_batch = tf.train.shuffle_batch([train_img,train_label], batch_size=batch_size, capacity=64,num_threads=8,min_after_dequeue=32)
    sess.run(tf.local_variables_initializer())
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # i = 0
    # while True:
    #     i += 1
    #     image_data = sess.run(img_train_batch)
    #     print(image_data)
    #     image_data=np.asarray(np.reshape(image_data[0:1,:,:,:],(360,480,3)),np.uint8)
    #     cv2.imshow("test",image_data)
    #     cv2.waitKey(1000)
    # coord.request_stop()
    # coord.join(threads)
    # train_image=tf.image.decode_png(train_img_file)
    # train_image = tf.image.resize_images(train_image, [height, width])
    # train_label = tf.image.resize_images(tf.image.decode_png(train_label_file), [height, width])
    # train_image = tf.cast(tf.reshape(train_image, [height, width, 3]), tf.float32)
    # train_label = tf.cast(tf.reshape(train_label, [height, width, 3]), tf.int64)
    # img_train_batch = tf.train.batch([train_image], batch_size=batch_size, capacity=64,num_threads=8)#min_after_dequeue=32,


    # '''test_data'''
    # test_img_queue = tf.train.string_input_producer(test_images_path, shuffle=False)
    # test_label_queue = tf.train.string_input_producer(test_labels_path, shuffle=False)
    # image_reader3 = tf.WholeFileReader()
    # image_reader4 = tf.WholeFileReader()
    # _, test_img_file = image_reader3.read(test_img_queue)
    # _, test_label_file = image_reader4.read(test_label_queue)
    # test_image = tf.image.resize_images(tf.image.decode_png(test_img_file), [height, width])
    # test_label = tf.image.resize_images(tf.image.decode_png(test_label_file), [height, width])
    # test_image =tf.cast(tf.reshape(test_image, [height, width, 3]), tf.float32)
    # test_label = tf.cast(tf.reshape(test_label, [height, width, 3]), tf.float32)
    # img_test_batch, label_test_batch = tf.train.shuffle_batch([test_image, test_label], batch_size=batch_size, capacity=32,
    #                                                             min_after_dequeue=20, num_threads=8)
    # sess.run(tf.local_variables_initializer())
    unet=U_Net(batch_size, width, height, channels, class_num=class_num, pretrained_model='resnet_v1_50.ckpt')
    unet.train_unet(sess, img_train_batch, label_train_batch,
                    epoch, train_image_num, learning_rate=0.005,
                    show_step=show_step)
    sess.close()

if __name__=='__main__':
    train()
    #forward("data/test","results")
