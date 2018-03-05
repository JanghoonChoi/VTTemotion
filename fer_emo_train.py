import sys,os,time
import numpy as np
import matplotlib.pyplot as plt
import cv2, pylab
import tensorflow as tf

fer_train_dict = np.load('db_dict/fer2013_train.npy').item()
fer_test_dict = np.load('db_dict/fer2013_test.npy').item()

net_chkpt_autoencode_path = 'net_chkpts_autoencode/'
net_chkpt_classify_path   = 'net_chkpts_classify/'

# labels
# 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral


# batch gen functions
def generate_fer2013_batch(fer_dict, batch_size, augment):
    key_list = fer_dict.keys()
    key_sel = np.random.choice(key_list, size=batch_size, replace=False)

    bat_im = np.zeros([batch_size, 48, 48, 1])
    bat_cl = np.zeros([batch_size, 7])

    for i in range(batch_size):
        img = np.reshape(fer_dict[key_sel[i]]['img']/255., [48,48,1])
        if augment:
            if np.random.rand() < 0.5:
                # horizontal flip
                img = np.flip(img, axis=1)
            
            if np.random.rand() < 0.5:
                # translation
                mod_x = np.random.randint(-1,2); mod_y = np.random.randint(-1,2)
                img = np.roll(img, mod_x, axis=1); img = np.roll(img, mod_y, axis=0)
            
            if np.random.rand() < 0.5:
                # noise
                img += 0.02*np.random.randn(48, 48, 1)

            if np.random.rand() < 0.5:
                # gamma
                img += (np.random.rand()-0.5)/5.

        # img to batch
        bat_im[i] = img
        # label to batch
        label = np.zeros([7])
        label[ fer_dict[key_sel[i]]['label'] ] = 1
        bat_cl[i] = label
    
    return bat_im, bat_cl

def fetch_fer2013_batch(fer_dict, start, end, augment):
    key_list = fer_dict.keys()
    set_size = np.size(key_list) 
    
    if end > set_size:
        end = set_size
    
    key_sel = key_list[start:end] #np.random.choice(key_list, size=batch_size, replace=False)
    batch_size = end-start

    bat_im = np.zeros([batch_size, 48, 48, 1])
    bat_cl = np.zeros([batch_size, 7])

    for i in range(batch_size):
        img = np.reshape(fer_dict[key_sel[i]]['img']/255., [48,48,1])
        if augment:
            if np.random.rand() < 0.5:
                # noise
                img += 0.02*np.random.randn(48, 48, 1)

            if np.random.rand() < 0.5:
                # gamma
                img += (np.random.rand()-0.5)/5.

        # img to batch
        bat_im[i] = img
        # label to batch
        label = np.zeros([7])
        label[ fer_dict[key_sel[i]]['label'] ] = 1
        bat_cl[i] = label
    
    return bat_im, bat_cl


# b_im, b_cl = generate_fer2013_batch(fer_train_dict, 1, True)

# plt.imshow(b_im[0,:,:,0], cmap='gray')
# print b_cl[0]


# In[ ]:


# params
do_train = 1
batch_size = 16
valid_size = 97
train_iter = 500000
learn_rate = 5e-5
train_conv = True
# output param
err_step = 50
val_step = 5*err_step
save_step= 500
std_dev  = 0.01
bn_decay = 0.99
dout_keepratio = 0.3
w_decay  = 1e-5

# init tf session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# def encoder vars
with tf.variable_scope("encoder") as scope:
    encoder_w = {
        'wc1': tf.get_variable('wc1', [5,5,1,32], initializer=tf.random_normal_initializer(stddev=std_dev), trainable=train_conv),
        'wc2': tf.get_variable('wc2', [5,5,32,64], initializer=tf.random_normal_initializer(stddev=std_dev), trainable=train_conv),
        'wc3': tf.get_variable('wc3', [3,3,64,128], initializer=tf.random_normal_initializer(stddev=std_dev), trainable=train_conv),
        'wc4': tf.get_variable('wc4', [3,3,128,256], initializer=tf.random_normal_initializer(stddev=std_dev), trainable=train_conv)
    }

#def encoder op
def encoder_net(in_img, w, decay, re):
    # conv1 - 24x24x32
    conv1 = tf.nn.conv2d(in_img, w['wc1'], strides=[1,2,2,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.contrib.layers.batch_norm(conv1, decay=decay, center=True,scale=True,trainable=train_conv,reuse=re, scope='encoder/wc1')
    
    # conv2 - 12x12x64
    conv2 = tf.nn.conv2d(conv1, w['wc2'], strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.max_pool(tf.nn.relu(conv2), ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, decay=decay, center=True,scale=True,trainable=train_conv,reuse=re, scope='encoder/wc2')

    # conv3 - 6x6x128
    conv3 = tf.nn.conv2d(conv2, w['wc3'], strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.max_pool(tf.nn.relu(conv3), ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3, decay=decay, center=True,scale=True,trainable=train_conv,reuse=re, scope='encoder/wc3')

    # conv4 - 3x3x256
    conv4 = tf.nn.conv2d(conv3, w['wc4'], strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.max_pool(tf.nn.relu(conv4), ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv4 = tf.contrib.layers.batch_norm(conv4, decay=decay, center=True,scale=True,trainable=train_conv,reuse=re, scope='encoder/wc4')
    
    return conv4

# # dummy placeholder
# _input_im  = tf.placeholder(tf.float32, shape=[None,48,48,1])
# _output_ft = encoder_net(_input_im, encoder_w, bn_decay, None)

# # restore encoder vars
# encoder_varlist = tf.global_variables()
# encoder_net_saver = tf.train.Saver()
# encoder_ckpt = tf.train.get_checkpoint_state(net_chkpt_autoencode_path)
# encoder_net_saver.restore(sess, encoder_ckpt.model_checkpoint_path)


# In[ ]:


# def classifier vars
with tf.variable_scope("classifier") as scope:
    classifier_w = {
        # kernel
        'fc1': tf.get_variable('fc1', [2304,1024], initializer=tf.random_normal_initializer(stddev=std_dev)),
        'fc2': tf.get_variable('fc2', [1024,7], initializer=tf.random_normal_initializer(stddev=std_dev)),
        # bias
        'bc1': tf.get_variable('bc1', [1024], initializer=tf.random_normal_initializer(stddev=std_dev)),
        'bc2': tf.get_variable('bc2', [7], initializer=tf.random_normal_initializer(stddev=std_dev))
    }

# def classifier op
def classifier_net(en_feat, w, keepratio):
    # fc1
    fc1 = tf.nn.relu(tf.add(tf.matmul(en_feat, w['fc1']), w['bc1']))
    fc1 = tf.nn.dropout(fc1, keepratio)
    # fc2
    fc2 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, w['fc2']), w['bc2']))
    # reshape
    out = tf.reshape(fc2, shape=[-1,7])
    
    return out
    
# def overall op
def emotion_net(in_img, w_encoder, w_classifier, decay, keepratio, re):
    en_feat = encoder_net(in_img, w_encoder, decay, re)
    en_feat = tf.reshape(en_feat, shape=[-1, 2304])
    out_class = classifier_net(en_feat, w_classifier, keepratio)
    
    return out_class


# def placeholders
input_im  = tf.placeholder(tf.float32, shape=[None,48,48,1])
output_cls = tf.placeholder(tf.float32, shape=[None,7])
keep_ratio = tf.placeholder(tf.float32)

# def model prediction
with tf.name_scope('model'):
    pred = emotion_net(input_im, encoder_w, classifier_w, bn_decay, keep_ratio, None)

# cross-entropy loss + l2 weight decay
err_l2_en = tf.multiply(w_decay, tf.add_n([tf.nn.l2_loss(wc) for wc in encoder_w.values()]) )
err_l2_cl = tf.multiply(w_decay, tf.add_n([tf.nn.l2_loss(wc) for wc in classifier_w.values()]) )
err_l2 = tf.add(err_l2_en, err_l2_cl)

err_ce = tf.losses.softmax_cross_entropy(output_cls, pred)

err = tf.add(err_ce, err_l2)
err_ratio = tf.div(err_ce, err_l2)

# optim
learn_rate_sch = tf.placeholder(tf.float64)
optm_0 = tf.train.AdamOptimizer(learning_rate=learn_rate_sch).minimize(err)

# init / saver
# init classifier vars
# classifier_varlist = [v for v in tf.global_variables() if encoder_varlist.count(v)==False]
# init = tf.variables_initializer(classifier_varlist)
init = tf.global_variables_initializer()
sess.run(init)
# new net saver
net_saver = tf.train.Saver(max_to_keep=100)



# In[ ]:


if do_train:
    # init text file
    today_dt = str(time.strftime("%m-%d-%H-%M", time.gmtime()))
    output_text = open("err_log/err_val_"+ today_dt + ".txt","w")
    output_text.close()
    
    # load ckpt
    ckpt = tf.train.get_checkpoint_state(net_chkpt_classify_path)
    if ckpt:
        net_saver.restore(sess, ckpt.model_checkpoint_path)
        print 'continue from checkpoint: '
        print ckpt
    else:
        print 'no checkpoint'
    
    # train loop
    for loop in range(train_iter):
        
        # generate train batch
        bat_im, bat_cls = generate_fer2013_batch(fer_train_dict, batch_size, True)
        
        sess.run(optm_0, feed_dict = {
            input_im:bat_im, output_cls:bat_cls, learn_rate_sch:learn_rate, keep_ratio:dout_keepratio
        })
        
        # valid err
        if loop % val_step == 0:
            valid = 0.
            accur = 0.
            valid_set_size = np.size(fer_test_dict.keys())
            val_times = int(np.ceil(valid_set_size / valid_size))
            
            for i in range(val_times):
                val_im, val_cls = fetch_fer2013_batch(fer_test_dict, valid_size*i, valid_size*(i+1), False)
                
                valid_elem, valid_pred = sess.run([err, pred], feed_dict={
                    input_im:val_im, output_cls:val_cls, keep_ratio:1.0
                })
                # for validation
                valid += (1./val_times)*valid_elem
                # for accuracy
                accur_elem = np.sum(np.array(val_cls.argmax(axis=1) == valid_pred.argmax(axis=1))) / float(valid_size)
                accur += (1./val_times)*accur_elem
                                
            print 'val: ' + str(valid)
            print 'accu: ' + str(accur)

            
        # err step, output
        if loop % err_step == 0:
            print 'loop: ' + str(loop)
            
            error, err_rat = sess.run([err, err_ratio], feed_dict={
                input_im:bat_im, output_cls:bat_cls, keep_ratio:dout_keepratio
            })
            print 'err: ' + str(error)
            print 'w/r: ' + str(err_rat)
            # open-write-close
            output_text = open("err_log/err_val_"+ today_dt + ".txt","a")
            output_text.write("%s, %s, %s\n" % (error, valid, accur))
            output_text.close()
        
        if loop % save_step == 0:
            net_saver.save(sess, net_chkpt_classify_path+'/chkpt-net.ckpt', global_step=loop)
            print "saved"

    
else:
    # test
    # load ckpt
    ckpt = tf.train.get_checkpoint_state(net_chkpt_classify_path)
    if ckpt:
        net_saver.restore(sess, ckpt.model_checkpoint_path)
        print 'continue from checkpoint: '
        print ckpt
    else:
        print 'no checkpoint'
    
    # generate batch
    bat_im, bat_cls = generate_fer2013_batch(fer_train_dict, batch_size, True)

    # run net
    bat_pred = sess.run(pred, feed_dict={input_im: bat_im})
    
    # display
    plt.imshow(bat_im[0,:,:,0],cmap='gray')
    print 'gt :' + str(bat_cls.argmax(axis=1)) + ', pred: ' + str(bat_pred.argmax(axis=1))
    print 'accur :' + str(np.sum(np.array(bat_cls.argmax(axis=1) == bat_pred.argmax(axis=1))) / float(batch_size))
#     f, ax = plt.subplots(1,3)
#     # f.set_figheight(6)
#     # f.set_figwidth(8)
#     ax[0].imshow(bat_in[0,:,:,0],cmap='gray',vmin=0.,vmax=1.)
#     ax[1].imshow(bat_out[0,:,:,0],cmap='gray',vmin=0.,vmax=1.)
#     ax[2].imshow(bat_pred[0,:,:,0],cmap='gray',vmin=0.,vmax=1.)
    

