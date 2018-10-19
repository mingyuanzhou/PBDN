#run this demo code to reproduce the results of PBDN-AIC-SGD and PBDN-AIC_{\epsilon=0.01}-SGD in Tables 2, 3, and 5.

#uncomment Line 571 (for i in np.array([16,17,18,19]):), comment Line 570 (for i in np.array([1,2,3,4,5,6,8,9]):), and then run the modified demo code to reproduce the results of PBDN in Table 1; run plot_subtype.m in Matlab to reproduce the subtype images in Table 1.

import numpy as np, scipy.sparse as sp
import scipy.io as sio
import math
import pdb
import tensorflow as tf
import matplotlib.pyplot as plt


import os
import urllib

import shutil



#if True:

#for JointLearn in np.array([False]):

def train_new_layer(y_,x_last_layer,depth,learning_rate,minibatchsize,datasize,\
                    W_side0,bb_side0,log_r_side0,log_gamma_side0,log_c_side0,K_side0,\
                    W_side1,bb_side1,log_r_side1,log_gamma_side1,log_c_side1,K_side1,\
                    a0,b0):

    layer1_side0   = tf.nn.softplus(tf.add(tf.matmul(x_last_layer, W_side0[depth]), bb_side0[depth]))
    log_1_p_side0 = -(tf.matmul(layer1_side0,tf.exp(log_r_side0))) #+0*tf.exp(br[side]))
    prob_side0 = -tf.expm1(log_1_p_side0)
    mask_true = tf.greater(y_-0.0,0.5)
    mask_false = tf.logical_not(mask_true)
    Loglike0  = tf.reduce_sum(tf.boolean_mask(log_1_p_side0,mask_false))\
                +tf.reduce_sum(tf.log(tf.boolean_mask(prob_side0,mask_true)))
                
    cross_entropy_side0 = 0 
    cross_entropy_side0 = cross_entropy_side0 -tf.reduce_sum((tf.exp(log_gamma_side0)/tf.cast(K_side0, tf.float32)-1)*log_r_side0-tf.exp(log_c_side0)*tf.exp(log_r_side0))/datasize
    cross_entropy_side0 = cross_entropy_side0  +(- (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(W_side0[depth])/(2*b0))) - (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(bb_side0[depth])/(2*b0))) )/datasize
                                        
    layer1_side1   = tf.nn.softplus(tf.add(tf.matmul(x_last_layer, W_side1[depth]), bb_side1[depth]))
    log_1_p_side1 = -(tf.matmul(layer1_side1,tf.exp(log_r_side1))) #+0*tf.exp(br[side]))
    prob_side1 = -tf.expm1(log_1_p_side1)
    Loglike1  = tf.reduce_sum(tf.boolean_mask(log_1_p_side1,mask_true))\
                +tf.reduce_sum(tf.log(tf.boolean_mask(prob_side1,mask_false)))
    
    cross_entropy_side1 = 0 
    cross_entropy_side1 = cross_entropy_side1-tf.reduce_sum((tf.exp(log_gamma_side1)/tf.cast(K_side1, tf.float32)-1)*log_r_side1-tf.exp(log_c_side1)*tf.exp(log_r_side1))/datasize
    cross_entropy_side1 = cross_entropy_side1 +(- (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(W_side1[depth])/(2*b0))) - (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(bb_side1[depth])/(2*b0))) )/datasize
                                           
    LogLike_combine = tf.reduce_sum(tf.log(tf.boolean_mask((1-prob_side0)/2.0+prob_side1/2.0,mask_false)))\
                +tf.reduce_sum(tf.log(tf.boolean_mask(prob_side0/2.0+(1-prob_side1)/2.0,mask_true)))
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_side0+cross_entropy_side1\
                                           -Loglike0/tf.cast(minibatchsize, tf.float32) -Loglike1/tf.cast(minibatchsize, tf.float32)  )
  
    return train_step,prob_side0,prob_side1, Loglike0, Loglike1, LogLike_combine


def next_batch(num, data, labels):

    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    labels_shuffle = np.reshape(labels_shuffle, (len(labels_shuffle), 1))
    return data_shuffle, labels_shuffle


def main(i,trial,dataname,Error_AIC, TT_AIC, Cost_AIC,Error_AIC_sparse, TT_AIC_sparse, Cost_AIC_sparse,fig): 
     
    if i<=7:
        content = sio.loadmat('data/benchmarks.mat');
        benchmark = content[dataname]  
        x_train = benchmark['x'][0,0][benchmark['train'][0,0][trial-1,:]-1,:]
        t_train = benchmark['t'][0,0][benchmark['train'][0,0][trial-1,:]-1]  
        t_train = np.reshape(t_train, (1,-1))[0]
        t_train [t_train ==-1]=0;
        x_test = benchmark['x'][0,0][benchmark['test'][0,0][trial-1,:]-1,:]
        t_test = benchmark['t'][0,0][benchmark['test'][0,0][trial-1,:]-1]  
        t_test = np.reshape(t_test, (1,-1))[0]
        t_test [t_test==-1] =0;  

    elif i==8:
        content = sio.loadmat('data/ijcnn1.mat')
        x_train = sp.csr_matrix(content['x_train'],dtype=np.float32)
        t_train = np.array(content['t_train'], dtype=np.int32) 
        t_train = np.reshape(t_train, (1,-1))[0]
        t_train [t_train ==-1]=0;
        x_test = sp.csr_matrix(content['x_test'],dtype=np.float32)
        x_test = sp.csr_matrix((x_test.data, x_test.indices, x_test.indptr), shape=(x_test.shape[0], x_train.shape[1]))
        t_test = np.array(content['t_test'], dtype=np.int32) 
        t_test = np.reshape(t_test, (1,-1))[0]
        t_test [t_test==-1] =0; 

        x = sp.vstack([x_train, x_test]).toarray()
        t = np.hstack([t_train,t_test])

        traindex = np.arange(trial-1,x.shape[0],10)
        testdex = np.arange(0,x.shape[0])
        testdex = np.delete(testdex,traindex)
        
        x_train=x[traindex,:]
        x_test=x[testdex,:]
        t_train=t[traindex]
        t_test=t[testdex]


    elif i==9:
        content = sio.loadmat('data/a9a.mat')
        
        x_train = sp.csr_matrix(content['x_train'],dtype=np.float32)
        t_train = np.array(content['t_train'], dtype=np.int32) 
        t_train = np.reshape(t_train, (1,-1))[0]
        t_train [t_train ==-1]=0;
        x_test = sp.csr_matrix(content['x_test'],dtype=np.float32)
        x_test = sp.csr_matrix((x_test.data, x_test.indices, x_test.indptr), shape=(x_test.shape[0], x_train.shape[1]))
        t_test = np.array(content['t_test'], dtype=np.int32) 
        t_test = np.reshape(t_test, (1,-1))[0]
        t_test [t_test==-1] =0; 

        x = sp.vstack([x_train, x_test]).toarray()
        t = np.hstack([t_train,t_test])

        traindex = np.arange(trial-1,x.shape[0],10)
        testdex = np.arange(0,x.shape[0])
        testdex = np.delete(testdex,traindex)
        
        x_train=x[traindex,:]
        x_test=x[testdex,:]
        t_train=t[traindex]
        t_test=t[testdex]

        x_train[0,np.sum(x_train,axis=0)==0] = np.finfo(np.float32).tiny
        x_test[0,np.sum(x_test,axis=0)==0] = np.finfo(np.float32).tiny
        
    elif i>=10:
        content = sio.loadmat('data/'+dataname+'.mat')   

        
        x_train = sp.csr_matrix(content['x_train'],dtype=np.float32)
        t_train = np.array(content['t_train'], dtype=np.int32) 
        t_train = np.reshape(t_train, (1,-1))[0]
        t_train [t_train ==-1]=0;
        x_test = sp.csr_matrix(content['x_test'],dtype=np.float32)
        x_test = sp.csr_matrix((x_test.data, x_test.indices, x_test.indptr), shape=(x_test.shape[0], x_train.shape[1]))
        t_test = np.array(content['t_test'], dtype=np.int32) 
        t_test = np.reshape(t_test, (1,-1))[0]
        t_test [t_test==-1] =0; 

        x = sp.vstack([x_train, x_test]).toarray()
        t = np.hstack([t_train,t_test])

        traindex = np.arange(trial-1,x.shape[0],10)
        testdex = np.arange(0,x.shape[0])
        testdex = np.delete(testdex,traindex)
        
        x_train=x[traindex,:]
        x_test=x[testdex,:]
        t_train=t[traindex]
        t_test=t[testdex]

        x_train[0,np.sum(x_train,axis=0)==0] = np.finfo(np.float32).tiny
        x_test[0,np.sum(x_test,axis=0)==0] = np.finfo(np.float32).tiny

    x_train_origin=x_train
    x_test_origin=x_test 
    t_train =t_train
    t_test=t_test
    t_train1= np.reshape(t_train, (len(t_train), 1))
    K_init = np.int32(np.round(10*np.log10(x_train_origin.shape[0])))


    #set model parameters   
    JointLearn=False  
    minibatchsize=100    
    learning_rate0=0.01;
    learning_rate=learning_rate0;
    a0=1e-6
    b0=1e-6  
        
    depth=-1
    flag=False

    Kadd=0;
    
    W_side0={}
    save_W_side0={}
    W_side1={}
    save_W_side1={}
    bb_side0={}
    save_bb_side0={}
    bb_side1={}
    save_bb_side1={}
    
    AICbreakFlag = False     
    AIC_sparsebreakFlag = False
    while True:
        depth=depth+1
        if flag:
            Kadd=Kadd+1
            learning_rate=learning_rate/2
            a0=a0*10;
            b0=b0*10;
            depth=depth-1
            x_train = x_train0
            x_test = x_test0
        else:
            Kadd=0;
            learning_rate=learning_rate0
            a0=1e-6
            b0=1e-6
            x_train0=x_train
            x_test0=x_test
        if depth==Depth:
            break
        

        print('Training Hidden Layer '+str(depth+1))
        print('Numerical error:'+str(flag))
      

        x = tf.placeholder(tf.float32, shape=[None,x_train_origin.shape[1]])
        y_ = tf.placeholder(tf.float32, [None, 1])


        K_side0=K_init
        K_side1=K_init
        if flag:
            K_side0 = K_side0+Kadd
            K_side1 = K_side0+Kadd
            
        cross_entropy_share=0.0
        x_last_layer = x
        layer_share_below_propogate  = x
                            
        for t in range(depth):
            if JointLearn==False:
                layer_share   = tf.concat([tf.nn.softplus(tf.add(tf.matmul(x_last_layer, save_W_side0[t]), save_bb_side0[t])),\
                                           tf.nn.softplus(tf.add(tf.matmul(x_last_layer, save_W_side1[t]), save_bb_side1[t]))],1)                       
                cross_entropy_share = cross_entropy_share + (- (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(save_W_side0[t])/(2*b0))) - (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(save_bb_side0[t])/(2*b0))))/datasize                        
                cross_entropy_share = cross_entropy_share + (- (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(save_W_side1[t])/(2*b0))) - (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(save_bb_side1[t])/(2*b0))))/datasize
            else:
                W_side0[t] = tf.Variable(save_W_side0[t])
                W_side1[t] = tf.Variable(save_W_side1[t])
                bb_side0[t] = tf.Variable(save_bb_side0[t])
                bb_side1[t] = tf.Variable(save_bb_side1[t])

                layer_share   = tf.concat([tf.nn.softplus(tf.add(tf.matmul(x_last_layer, W_side0[t]), bb_side0[t])),\
                                           tf.nn.softplus(tf.add(tf.matmul(x_last_layer, W_side1[t]), bb_side1[t]))],1)  
                cross_entropy_share = cross_entropy_share + (- (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(W_side0[t])/(2*b0))) - (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(bb_side0[t])/(2*b0))))/datasize                        
                cross_entropy_share = cross_entropy_share + (- (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(W_side1[t])/(2*b0))) - (-a0-1/2)*tf.reduce_sum(tf.log1p(tf.square(bb_side1[t])/(2*b0))))/datasize
            #x_last_layer = layer_share
            #layer_share = tf.log(tf.maximum(layer_share,np.finfo(np.float32).tiny))
            #x_last_layer = tf.concat([layer_share,tf.nn.softplus(layer_share_below_propogate)],1)
            x_last_layer = tf.concat([layer_share,layer_share_below_propogate],1)
            #x_last_layer = tf.log(tf.maximum(x_last_layer,np.finfo(np.float32).tiny))
            #x_last_layer = tf.log(tf.maximum(layer_share,np.finfo(np.float32).tiny))
            layer_share_below_propogate = layer_share
      
        
        W_side0[depth] = tf.Variable(tf.random_normal([x_last_layer.shape[1].value, K_side0])/10)
        bb_side0[depth] = tf.Variable(tf.random_normal([1,K_side0])/10)
        log_r_side0 = tf.Variable(tf.random_normal([K_side0,1])/10)
            
        W_side1[depth] = tf.Variable(tf.random_normal([x_last_layer.shape[1].value, K_side1])/10)
        bb_side1[depth] = tf.Variable(tf.random_normal([1,K_side1])/10)
        log_r_side1 = tf.Variable(tf.random_normal([K_side1,1])/10)                       
              
        log_gamma_side0=tf.cast(tf.zeros([1])+tf.log(1.0), tf.float32)
        log_c_side0=tf.cast(tf.zeros([1]), tf.float32)
        log_gamma_side1=tf.cast(tf.zeros([1])+tf.log(1.0), tf.float32)
        log_c_side1=tf.cast(tf.zeros([1]), tf.float32)
           
        datasize = tf.cast(x_train.shape[0], tf.float32)       
        train_step,prob_side0,prob_side1, Loglike0, Loglik1, LogLike_combine = train_new_layer(y_,x_last_layer,depth,learning_rate,minibatchsize,datasize,\
                                                                                                W_side0,bb_side0,log_r_side0,log_gamma_side0,log_c_side0,K_side0,\
                                                                                                W_side1,bb_side1,log_r_side1,log_gamma_side1,log_c_side1,K_side1,\
                                                                                                a0,b0)
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        x_train = sess.run(x_last_layer,feed_dict={x: x_train_origin, y_: t_train1})
        if depth==0:
            num_batch = 4000
            learning_rate=0.01
        else:
            num_batch = 4000  
            learning_rate = learning_rate=0.05/(5.0+depth)
        #tic()
        for batch in range(num_batch):
            batch_xs, batch_ys = next_batch(minibatchsize,x_train_origin,t_train)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys-0.0})
            
            
            if (batch % 500 == 1) and (batch>500):

                #tic()
                p_ik   = tf.nn.softplus(tf.add(tf.matmul(x_train, W_side0[depth]), bb_side0[depth]))
                p_ik = -tf.expm1(-(tf.multiply(p_ik,tf.transpose(tf.exp(log_r_side0)))))
                b_ik = tf.cast(tf.greater(p_ik,tf.random_uniform(p_ik.shape)),tf.float32)
                
                
                b_i = (tf.logical_and(tf.greater(t_train+0.0,0.5),tf.greater(0.5,tf.reduce_sum(b_ik,1)))).eval()
                temp = tf.boolean_mask(p_ik,b_i);
                temp = tf.cumsum(temp,axis=1).eval()
                temp = tf.reduce_sum(tf.cast(tf.greater(tf.multiply(tf.reshape(temp[:,K_side0-1],[-1,1]),tf.random_uniform([temp.shape[0],1])),temp),tf.int32),1).eval()
                
                row=np.transpose(tf.where(b_i).eval())[0]
                col=temp
                b_ik = b_ik + tf.cast(sp.csr_matrix( (np.ones(temp.shape[0]),(row,col)), shape=(b_ik.shape[0].value,b_ik.shape[1].value) ).todense(),tf.float32)
                
                
                b_k = tf.greater(tf.reduce_sum(tf.cast(b_ik, tf.float32),0),0.5).eval()
                #K_side0 = tf.reduce_sum(tf.cast(b_k, tf.int32),0)
                W0 = tf.cast(tf.transpose( tf.boolean_mask(tf.transpose(W_side0[depth]),b_k)).eval(),tf.float32)
                r0 = tf.cast(tf.boolean_mask(tf.exp(log_r_side0),b_k).eval(),tf.float32)
                bb0 = tf.cast(tf.transpose(tf.boolean_mask(tf.transpose(bb_side0[depth]),b_k)).eval(),tf.float32)
                #toc()

                
                
                #tic()
                p_ik   = tf.nn.softplus(tf.add(tf.matmul(x_train, W_side1[depth]), bb_side1[depth]))
                p_ik = -tf.expm1(-(tf.multiply(p_ik,tf.transpose(tf.exp(log_r_side1)))))
                b_ik = tf.cast(tf.greater(p_ik,tf.random_uniform(p_ik.shape)),tf.float32)
                
                
                b_i = (tf.logical_and(tf.greater(1.0-t_train+0.0,0.5),tf.greater(0.5,tf.reduce_sum(b_ik,1)))).eval()
                temp = tf.boolean_mask(p_ik,b_i);
                temp = tf.cumsum(temp,axis=1).eval()
                temp = tf.reduce_sum(tf.cast(tf.greater(tf.multiply(tf.reshape(temp[:,K_side1-1],[-1,1]),tf.random_uniform([temp.shape[0],1])),temp),tf.int32),1).eval()
                
                row=np.transpose(tf.where(b_i).eval())[0]
                col=temp
                b_ik = b_ik + tf.cast(sp.csr_matrix( (np.ones(temp.shape[0]),(row,col)), shape=(b_ik.shape[0].value,b_ik.shape[1].value) ).todense(),tf.float32)
                
                 
                b_k = tf.greater(tf.reduce_sum(tf.cast(b_ik, tf.float32),0),0.5).eval()
                #K_side0 = tf.reduce_sum(tf.cast(b_k, tf.int32),0)
                W1 = tf.cast(tf.transpose( tf.boolean_mask(tf.transpose(W_side1[depth]),b_k)).eval(),tf.float32)
                r1 = tf.cast(tf.boolean_mask(tf.exp(log_r_side1),b_k).eval(),tf.float32)
                bb1 = tf.cast(tf.transpose(tf.boolean_mask(tf.transpose(bb_side1[depth]),b_k)).eval(),tf.float32)
                #toc()
                


                sess.close()

                K_side0 = W0.shape[1].value+0
                K_side1 = W1.shape[1].value+0                

                memory()
                #print([batch,rrr0[1],rrr1[1]])
                if bb0.shape[0].value>0: # W0.shape[1].value>0:
                    W_side0[depth] = tf.Variable(W0)
                    bb_side0[depth] = tf.Variable(bb0)
                    log_r_side0 = tf.Variable(tf.log(r0))
                else:
                    W_side0[depth] = tf.Variable(tf.random_normal([x_train.shape[1], K_side0])/10)
                    bb_side0[depth] = tf.Variable(tf.random_normal([1,K_side0])/10)
                    log_r_side0 = tf.Variable(tf.random_normal([K_side0,1])/10)
                if bb1.shape[0].value>0:  #W1.shape[1].value>0:    
                    W_side1[depth] = tf.Variable(W1)
                    bb_side1[depth] = tf.Variable(bb1)
                    log_r_side1 = tf.Variable(tf.log(r1))
                else:
                    W_side1[depth] = tf.Variable(tf.random_normal([x_train.shape[1], K_side1])/10)
                    bb_side1[depth] = tf.Variable(tf.random_normal([1,K_side1])/10)
                    log_r_side1 = tf.Variable(tf.random_normal([K_side1,1])/10)
                
                train_step,prob_side0,prob_side1, Loglike0, Loglike1, LogLike_combine = train_new_layer(y_,x_last_layer,depth,learning_rate,minibatchsize,datasize,\
                                                                                                W_side0,bb_side0,log_r_side0,log_gamma_side0,log_c_side0,K_side0,\
                                                                                                W_side1,bb_side1,log_r_side1,log_gamma_side1,log_c_side1,K_side1,\
                                                                                                a0,b0)
                sess = tf.InteractiveSession()
                tf.global_variables_initializer().run()
                
        #toc()
        if math.isnan((tf.reduce_sum(log_r_side0)+tf.reduce_sum(log_r_side1)+tf.reduce_sum(W_side0[depth])+tf.reduce_sum(W_side1[depth])+tf.reduce_sum(bb_side0[depth])+tf.reduce_sum(bb_side1[depth])).eval()):
            flag=True
            break
        else:
            flag=False
        
        # Test trained model
        correct_prediction = tf.equal(tf.greater(prob_side0,prob_side1), tf.greater(y_-0.0,0.5))
        accuracy_score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        t_test1= np.reshape(t_test, (len(t_test), 1))

        accuracy = sess.run(accuracy_score, feed_dict={x: x_test_origin, y_: t_test1})

        Errors[i-1,trial-1,depth]=1-accuracy

        print(dataname+'_trial'+str(trial)+'_depth'+str(depth+1)+'_'+str(Errors[i-1,trial-1,depth]))
 
        
        save_W_side0[depth] =tf.constant(W_side0[depth].eval())
        save_W_side1[depth]=tf.constant((W_side1[depth]).eval())
        save_bb_side0[depth] =tf.constant((bb_side0[depth]).eval())
        save_bb_side1[depth] =tf.constant((bb_side1[depth]).eval())
        if JointLearn==True:    
            for t in range(depth):
                save_W_side0[t] =tf.constant(W_side0[t].eval())
                save_W_side1[t] =tf.constant(W_side1[t].eval())
                save_bb_side0[t] =tf.constant(bb_side0[t].eval())
                save_bb_side1[t] =tf.constant(bb_side1[t].eval())
                
        
        KKK_side0[i-1,trial-1,depth]=K_side0
        KKK_side1[i-1,trial-1,depth]=K_side1

        
        #Train_loglike = Temp
        Train_loglike=np.array([0,0])
        Train_loglike[0],Train_loglike[1], LogLike_combine= sess.run([Loglike0,Loglike1,LogLike_combine], feed_dict={x: x_train_origin, y_: t_train1})
        #Train_loglike_combine = sess.run(ogLike_combine, feed_dict={x: x_train_origin, y_: t_train1})
        
        Train_loglike_side0[i-1,trial-1,depth]=Train_loglike[0]
        Train_loglike_side1[i-1,trial-1,depth]=Train_loglike[1]
        
        aic=0.0
        aic_sparse=0.0
        cost = 0.0
        for t in range(depth+1):
            if t==0:
                K0 = tf.shape(save_W_side0[0]).eval()[0]
            else:
                K0 = KKK_side0[i-1,trial-1,t-1] + KKK_side1[i-1,trial-1,t-1]
                aic = aic-2*K0
            aic = aic + 2*(K0+2)*(KKK_side0[i-1,trial-1,t] + KKK_side1[i-1,trial-1,t])
            if t>0:
                aic_sparse = aic_sparse - 2*K0       
            sparse_threshold = 0.01
            temp1= np.vstack((save_W_side0[t].eval(), save_bb_side0[t].eval()))
            temp2= np.vstack((save_W_side1[t].eval(), save_bb_side1[t].eval()))
            aic_sparse = aic_sparse +2*np.count_nonzero(abs(temp1)>sparse_threshold*np.amax(abs(temp1)))\
                                    +2*np.count_nonzero(abs(temp2)>sparse_threshold*np.amax(abs(temp2)))\
                                    +2*tf.shape(save_W_side0[t]).eval()[1]+2*tf.shape(save_W_side1[t]).eval()[1];
            cost= cost+np.size(temp1)+np.size(temp2)
        cost = cost/(tf.shape(save_W_side0[0]).eval()[0]+1.0)
        aic = aic-2*Train_loglike[0]-2*Train_loglike[1]
        aic_sparse = aic_sparse-2*Train_loglike[0]-2*Train_loglike[1]
        AIC[i-1,trial-1,depth]=aic
        AIC_sparse[i-1,trial-1,depth]=aic_sparse
        Cost[i-1,trial-1,depth] = cost
        if depth==0:
            AIC_min = np.inf
        if aic<AIC_min:
            AIC_min=aic
            Error_AIC[i-1,trial-1] =  Errors[i-1,trial-1,depth]
            TT_AIC[i-1,trial-1] =  depth+1
            Cost_AIC[i-1,trial-1] = Cost[i-1,trial-1,depth]
        else:
            AIC_min = -np.inf
            AICbreakFlag = True
            
        if depth==0:
            AIC_sparse_min = np.inf
        if aic_sparse<AIC_sparse_min :
            AIC_sparse_min=aic_sparse
            Error_AIC_sparse[i-1,trial-1] =  Errors[i-1,trial-1,depth]
            TT_AIC_sparse[i-1,trial-1] =  depth+1
            Cost_AIC_sparse[i-1,trial-1] = Cost[i-1,trial-1,depth]
        else:
            AIC_sparse_min  = -np.inf   
            AIC_sparsebreakFlag = True
        print(dataname+'_trial'+str(trial)+'_LogLike'+str(depth+1)+'_'+str(Train_loglike[0])+','+str(Train_loglike[1])+','+str(LogLike_combine))
        print(dataname+'_trial'+str(trial)+'_K'+str(depth+1)+'_'+str(KKK_side0[i-1,trial-1,t])+','+str(KKK_side1[i-1,trial-1,t]))
        print(dataname+'_trial'+str(trial)+'_AIC'+str(depth+1)+'_'+str(AIC[i-1,trial-1,depth]))
        print(dataname+'_trial'+str(trial)+'_AICsparse'+str(depth+1)+'_'+str(AIC_sparse[i-1,trial-1,depth]))
        print(dataname+'_trial'+str(trial)+'_ErrorAIC'+'_'+str(Error_AIC[i-1,trial-1] )+'_TT'+'_'+str(TT_AIC[i-1,trial-1] ))
        print(dataname+'_trial'+str(trial)+'_ErrorAIC_sparse'+'_'+str(Error_AIC_sparse[i-1,trial-1])+'_TT'+'_'+str(TT_AIC_sparse[i-1,trial-1] ))
        print('************************')
        

        if (AICbreakFlag and AIC_sparsebreakFlag) or (depth==Depth-1):
            depth0=depth
            if (depth==Depth-1) and (not AICbreakFlag):
                depth0=Depth    
            for t in range(depth0):
                #print('Size of layer' +str(t+1)+': '+ str(tf.shape(save_W_side0[t]).eval()[0])+' * (' + str(tf.shape(save_W_side0[t]).eval()[1])+','+ str(tf.shape(save_W_side1[t]).eval()[1])+')')
                if t==0:
                    print('Size of layer' +str(t+1)+': '+ str(x_train_origin.shape[1])+' * (' + str(KKK_side0[i-1,trial-1,t])+','+ str(KKK_side1[i-1,trial-1,t])+')')
                elif t==1:
                    print('Size of layer' +str(t+1)+': '+ str( #x_train_origin.shape[1]+
                          KKK_side0[i-1,trial-1,t-1]+KKK_side1[i-1,trial-1,t-1])+' * (' + str(KKK_side0[i-1,trial-1,t])+','+ str(KKK_side1[i-1,trial-1,t])+')')
                else:
                    print('Size of layer' +str(t+1)+': '+ str( #KKK_side0[i-1,trial-1,t-2]+KKK_side1[i-1,trial-1,t-2]+
                          KKK_side0[i-1,trial-1,t-1]+KKK_side1[i-1,trial-1,t-1])+' * (' + str(KKK_side0[i-1,trial-1,t])+','+ str(KKK_side1[i-1,trial-1,t])+')')
            sess.close()
            #return Error_AIC[i-1,trial-1], TT_AIC[i-1,trial-1], Cost_AIC[i-1,trial-1]
            return Error_AIC, TT_AIC, Cost_AIC,Error_AIC_sparse, TT_AIC_sparse, Cost_AIC_sparse,fig

  
        WWW0=[]
        BBB0=[]
        WWW1=[]
        BBB1=[]
        for t in range(depth+1):
            if t==0:
                WWW0=[WWW0,save_W_side0[t].eval()]
                BBB0=[BBB0,save_bb_side0[t].eval()]
                WWW1=[WWW1,save_W_side1[t].eval()]
                BBB1=[BBB1,save_bb_side1[t].eval()]
            else:
                WWW0.append(save_W_side0[t].eval())
                BBB0.append(save_bb_side0[t].eval())
                WWW1.append(save_W_side1[t].eval())
                BBB1.append(save_bb_side1[t].eval())
        sio.savemat(dataname+'_PBDN_para'+ '.mat', {'Errors':Errors,'KKK_side0':KKK_side0,'KKK_side1':KKK_side1,\
                    'Train_loglike_side0':Train_loglike_side0,'Train_loglike_side1':Train_loglike_side1,\
                    'r_side0':np.exp(log_r_side0.eval()), 'r_side1':np.exp(log_r_side1.eval()),\
                    'AIC':AIC,'AIC_sparse':AIC_sparse,'Cost':Cost,\
                    'Error_AIC':Error_AIC,'TT_AIC':TT_AIC,'Cost_AIC':Cost_AIC,\
                    'Error_AIC_sparse':Error_AIC_sparse,'TT_AIC_sparse':TT_AIC_sparse,'Cost_AIC_sparse':Cost_AIC_sparse,\
                    'WWW0':WWW0, 'WWW1':WWW1,'BBB0':BBB0, 'BBB1':BBB1})        
        sess.close()
        if AICbreakFlag and AIC_sparsebreakFlag:            
            break 
        
    print('###############################')
    print('###############################'+dataname+'_trial'+str(trial)+'_Error_combine'+'_'+str(Error_AIC[i-1,trial-1]))
    print('###############################')
   
               
if __name__ == "__main__":
    datanames = ['banana', #1
    'breast_cancer', #2
    'titanic', #3
    'waveform', #4
    'german', #5
    'image', #6
    'pima_diabetes', #7
    'ijcnn1', #8
    'a9a', #9
    'diabetis', #10
    'circle', #11
    'xor', #12
    'dbmoon', #13
    'USPS3v5', #14
    'mnist2vother', #15
    'mnist3v5', #16
    'mnist3v8', #17
    'mnist4v7', #18
    'mnist4v9' #19
    ];
    Depth=5
    Errors = np.zeros([19,10,Depth])
    KKK_side0 = np.zeros([19,10,Depth])
    KKK_side1 = np.zeros([19,10,Depth])
    Train_loglike_side0 = np.zeros([19,10,Depth])
    Train_loglike_side1 = np.zeros([19,10,Depth])
    AIC = np.zeros([19,10,Depth])
    AIC_sparse = np.zeros([19,10,Depth])
    
    Cost = np.zeros([19,10,Depth])
    
    Error_AIC =  np.zeros([19,10])
    TT_AIC =  np.zeros([19,10])
    Cost_AIC =  np.zeros([19,10])
    Error_AIC_sparse =  np.zeros([19,10])
    TT_AIC_sparse =  np.zeros([19,10])
    Cost_AIC_sparse =  np.zeros([19,10])
    

    for i in np.array([1,2,3,4,5,6,8,9]):
    #for i in np.array([16,17,18,19]):
        if i<=6:
            maxTrial=10
        else:
            maxTrial=5
        #maxTrial=5;
        Depth=5
        fig, axarr = plt.subplots(Depth,maxTrial,figsize=(30, 15))
        #fig=0
        dataname = datanames[i-1]
        
        def memory():
            import os
            import psutil
            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
            print('memory use:', memoryUse)
        def tic():
            #Homemade version of matlab tic and toc functions
            import time
            global startTime_for_tictoc
            startTime_for_tictoc = time.time()
    
        def toc():
            import time
            if 'startTime_for_tictoc' in globals():
                print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
            else:
                print "Toc: start time not set"
        
        
        for trial in range(1,maxTrial+1):
            with tf.Graph().as_default():
                tic()
                Error_AIC, TT_AIC, Cost_AIC, Error_AIC_sparse, TT_AIC_sparse, Cost_AIC_sparse, fig = main(i,trial,dataname,Error_AIC, TT_AIC, Cost_AIC,Error_AIC_sparse, TT_AIC_sparse, Cost_AIC_sparse,fig)
                fig.savefig(dataname+'_PBDN'+'.pdf')   # save the figure to file
                memory()
                print('###############################')
                print('###############################'+dataname+'_trial'+str(trial)+'_Error_combine'+'_'+str(Error_AIC[i-1,trial-1]))
                print('###############################')
            
                sio.savemat(dataname+'_PBDN_results.mat', {'Error_AIC':Error_AIC,'TT_AIC':TT_AIC,'Cost_AIC':Cost_AIC,'Error_AIC_sparse':Error_AIC_sparse,'TT_AIC_sparse':TT_AIC_sparse,'Cost_AIC_sparse':Cost_AIC_sparse})        
                toc()
        plt.close(fig) 
    #eng.quit()       
