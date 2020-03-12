import tensorflow as tf
from tqdm import tqdm
from model import generator, discriminor, softmax, presoftmax

def wgan(att, train, unseen, seen, opt):
    # 配置 #
    tf.random.set_seed(opt.random_seed)      # random_seed
    # 配置 #
    
    # 数据读取 #
    train_X = tf.transpose(train[0])
    train_y = train[1]
    train_label, train_inx = tf.unique(train_y)
    train_onehot = tf.one_hot(train_inx, depth=train_label.shape[0])
    cps_db = tf.data.Dataset.from_tensor_slices((train_X, train_onehot))
    cps_db = cps_db.shuffle(train_X.shape[0]).batch(opt.class_batch)

    train_att = tf.transpose(tf.gather(att, train_y-1, axis=1))
    train_db = tf.data.Dataset.from_tensor_slices((train_X, train_onehot, train_att))
    train_db = train_db.shuffle(train_X.shape[0]).batch(opt.train_batch)

    unseen_y = unseen[1]
    unseen_labels, _ = tf.unique(unseen_y)
    unseen_att = tf.gather(att, unseen_labels-1, axis=1)
    unseen_att = tf.transpose(unseen_att)
    # 数据读取 #

    # WGAN-定义 #
    generate = generator()
    discrim = discriminor()
    unseen_y = unseen[1]
    unseen_label, _ = tf.unique(unseen_y)
    class_test = softmax(unseen_label.shape[0], opt.class_regularizer)  # 参数是不可见类的类数量
    class_all_test = softmax(att.shape[1], opt.class_regularizer)

    zc_optimizer = tf.keras.optimizers.Nadam(opt.class_lr, beta_1=0.5)
    gzc_optimizer = tf.keras.optimizers.Nadam(opt.class_lr, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Nadam(opt.train_lr, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Nadam(opt.train_lr, beta_1=0.5)

    max_zsl = 0
    max_gu = 0
    max_gs = 0
    max_h = 0
    # WGAN-定义 #


    # 预训练分类器 #
    cps = pretrain(cps_db, train_label.shape[0], opt.pre_epoch, opt.pre_class_lr, opt.pre_classifier_read, opt)
    # 预训练分类器 #

    # 训练 #
    for epoch in range(opt.train_epochs):
        print("第", epoch+1, "次迭代:")
        print("生成器和判别器训练:")

        # 训练GAN网路 #
        for _, (x_b, y_b, att_b) in tqdm(enumerate(train_db)):
            for _ in range(5):  # 判别器训练      
                noise = tf.random.truncated_normal(att_b.shape)
                g_x = generate.call(att_b, noise)

                with tf.GradientTape() as tape1:
                    wd_real = tf.reduce_mean(discrim.call(x_b, att_b))
                    wd_fake = tf.reduce_mean(discrim.call(g_x, att_b))
                    gp = gradient_penalty(discrim, x_b, g_x, att_b)

                    loss_d = -wd_real + wd_fake + opt.gp_lambda * gp 
                grads = tape1.gradient(loss_d, discrim.trainable_variables)
                d_optimizer.apply_gradients(zip(grads, discrim.trainable_variables))                           
            
            with tf.GradientTape() as tape2: # 生成器训练 
                noise = tf.random.truncated_normal(att_b.shape)
                g_x = generate.call(att_b, noise)

                wd_fake = tf.reduce_mean(discrim.call(g_x, att_b))
                pre_label = cps.call(g_x)
                loss_c = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_b, pre_label, from_logits=False))

                loss_g = -wd_fake +  loss_c * opt.cls_beita
            grads = tape2.gradient(loss_g, generate.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generate.trainable_variables))
        print("第%d次迭代:D->%f, G->%f"%(epoch+1, loss_d, loss_g))
        # 训练GAN网路 #

        # 生成数据 #
        generate_x, generate_y = syn_features(generate, unseen, att, opt.generate_num)   # 生成数据
        generate_label, generate_inx = tf.unique(generate_y)
        generate_onehot = tf.one_hot(generate_inx, depth=generate_label.shape[0])
        generate_db = tf.data.Dataset.from_tensor_slices((generate_x, generate_onehot))
        generate_db = generate_db.shuffle(generate_x.shape[0]).batch(opt.class_batch)

        all_x = tf.concat([train_X, generate_x], axis=0)
        all_y = tf.concat([train_y, generate_y], axis=0)
        all_label, all_inx = tf.unique(all_y)  

        mask = [0] * all_label.shape[0]
        for label in train_label:
            loc = tf.where(all_label == label)
            mask[loc[0][0]] += 1
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        all_onehot = tf.one_hot(all_inx, depth=all_label.shape[0])
        all_db = tf.data.Dataset.from_tensor_slices((all_x, all_onehot))
        all_db = all_db.shuffle(all_x.shape[0]).batch(opt.class_batch)
        # 生成数据

        print("分类器训练:")
        mid_zsl = 0
        mid_H = 0
        mid_gu = 0
        mid_gs = 0
        

        # 训练分类器并计算准确度 # 
        for _ in tqdm(range(opt.valid_epoch)):
            for _, (test_x, test_y) in enumerate(generate_db):  # zsl分类器训练
                with tf.GradientTape() as tape3: # 生成器训练
                    pre_y = class_test(test_x)
                    loss_regular = tf.add_n(class_test.losses)
                    loss_zc = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(test_y, pre_y, from_logits=False)) + loss_regular
                grads = tape3.gradient(loss_zc, class_test.trainable_variables)
                zc_optimizer.apply_gradients(zip(grads, class_test.trainable_variables))               

            for _, (all_x, all_y) in enumerate(all_db):     # gzsl分类器训练
                with tf.GradientTape() as tape4: # 生成器训练
                    pre_y = class_all_test(all_x)
                    loss_regular = tf.add_n(class_all_test.losses)
                    loss_gzc = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(all_y, pre_y, from_logits=False)) + loss_regular
                grads = tape4.gradient(loss_gzc, class_all_test.trainable_variables)
                gzc_optimizer.apply_gradients(zip(grads, class_all_test.trainable_variables))
            
            details = caculate(class_test, unseen, generate_label)
            ac_unseen = tf.reduce_mean(details)
            gdetails = caculate(class_all_test, unseen, all_label, True, opt.calibrate, mask)
            gac_unseen = tf.reduce_mean(gdetails)
            gac_seen = caculate(class_all_test, seen, all_label, True, opt.calibrate, mask)
            gac_seen = tf.reduce_mean(gac_seen)
            H = (2*gac_unseen*gac_seen) / (gac_seen + gac_unseen)
            if ac_unseen > mid_zsl:
                mid_zsl = ac_unseen
            if H > mid_H:
                mid_H = H
                mid_gu = gac_unseen
                mid_gs = gac_seen           

        if mid_zsl > max_zsl:
            max_zsl = mid_zsl
            generate.save_weights('ckpt/generator.ckpt')
            discrim.save_weights('ckpt/discriminor.ckpt')
        if mid_H > max_h:
            max_h = mid_H
            max_gu = mid_gu
            max_gs = mid_gs
            generate.save_weights('ckpt/gzsl_generate.ckpt')
            discrim.save_weights('ckpt/gzsl_discrim.ckpt')
    
        print("第%d次迭代,ZSL精确度为:%f" %(epoch+1, mid_zsl * 100))         
        print("第%d次迭代,GZSL不可见类精确度为:%f" %(epoch+1, mid_gu * 100))
        print("第%d次迭代,GZSL可见类精确度为:%f" %(epoch+1, mid_gs * 100))
        print("第%d次迭代,GZSL综合指标精确度为:%f" %(epoch+1, mid_H * 100))
        print("当前最大:ZSL->%f, GZSL-unseen->%f, GZSL-seen->%f, GZSL-H->%f" %(max_zsl*100, max_gu*100, max_gs*100, max_h*100))
        print("\n")   
        # 训练分类器并计算准确度 #      
    # 训练 #

    return (max_zsl, max_gu, max_gs, max_h)


def pretrain(data, class_num, pre_epoch, lr, ifread, opt):
    cps = presoftmax(class_num)
    if ifread:
        cps.load_weights('ckpt/cps.ckpt').expect_partial()
    else:
        cps.compile(tf.keras.optimizers.Nadam(lr, beta_1=0.5),
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
        cps.fit(data, epochs=pre_epoch)
        cps.save_weights('ckpt/cps.ckpt')
    return cps


def gradient_penalty(discriminator, batch_x, fake_image, batch_s):      # 计算GP值
    batchsz = batch_x.shape[0]
    t = tf.random.uniform([batchsz, 1])
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image
    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, batch_s)
    grads = tape.gradient(d_interplote_logits, interplate)
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp-1) ** 2)

    return gp

def syn_features(generate, unseen, att, generate_num):    # 生成数据
    unseen_y = unseen[1]
    unseen_label, _ = tf.unique(unseen_y)
    unseen_att = tf.transpose(tf.gather(att, unseen_label-1, axis=1))
    generate_x = []
    generate_y = []
    for k in range(unseen_att.shape[0]):
        curr = tf.reshape(unseen_att[k], [1, -1])
        curr = tf.broadcast_to(curr, [generate_num, unseen_att.shape[1]])       # 生成平均点改动处
        noise  = tf.random.truncated_normal(curr.shape)
        generate_mid = generate.call(curr, noise)
        generate_x.append(generate_mid)
        generate_y.append(tf.fill([generate_num], unseen_label[k]))
    dim = generate_mid.shape[-1]
    generate_x = tf.reshape(generate_x, [-1, dim])
    generate_y = tf.reshape(generate_y, [-1])
    return generate_x, generate_y

def caculate(model, data, label_log, ifgzsl=False, calibrate=0, mask=None):           # 计算准确度
    test_x = tf.transpose(data[0])
    test_y = data[1]
    label, inx = tf.unique(test_y)
    num = [0] * label.shape[0]
    count = [0] * label.shape[0]
    pre_y = model(test_x)
    if ifgzsl:
        calibrate_factor = mask * calibrate
        calibrate_factor = tf.broadcast_to(calibrate_factor, [test_x.shape[0], pre_y.shape[1]])
        pre_y = pre_y - calibrate_factor
    pre_inx = tf.argmax(pre_y, axis=1)
    pre_label = tf.gather(label_log, pre_inx)
    for i in range(len(num)):
        mask = inx == i
        sta = tf.cast(mask, tf.int32)
        num[i] += int(tf.reduce_sum(sta))
        curr_label = label[i]
        curr_pre_label = pre_label[mask]
        mask_label = curr_pre_label == curr_label
        mask_label = tf.cast(mask_label, tf.int32)
        count[i] += int(tf.reduce_sum(mask_label))
    num = tf.convert_to_tensor(num)
    count = tf.convert_to_tensor(count)
    return count/num
