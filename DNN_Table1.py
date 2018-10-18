#Reproduce the results of DNNs in Table 1

import numpy as np, scipy.sparse as sp
#from svmutil import *
import scipy.io as sio

import pdb
import tensorflow as tf



import os
import urllib

import shutil



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

def main():
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
    Errors = np.zeros([19,10])
    #for i in np.array([1,2,3,4,5,6,8,9]): 
    for i in np.array([16,17,18,19]): #[1,2,3,4,5,6,8,9]): 
        if i<=6:
            maxTrial=10
        else:
            maxTrial=5

        for trial in range(1,maxTrial+1):
            dataname = datanames[i-1];
            if i<=6:
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

                #x_train[0,np.sum(x_train,axis=0)==0] = np.finfo(np.float32).tiny
                #x_test[0,np.sum(x_test,axis=0)==0] = np.finfo(np.float32).tiny

                

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
                #content = sio.loadmat('data/USPS3v5.mat')
                #content = sio.loadmat('data/mnist4v9.mat')  
                content = sio.loadmat('data/'+dataname+'.mat')   
                #content = sio.loadmat('data/mnist2vNo2.mat')   
                #content = sio.loadmat('data/usps4vother.mat')
                #dataname='mnist4v7'
                #datanames[i-1]=dataname
                
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
                


             
            tic()

            # Specify that all features have real-value data
            #pdb.set_trace()
            feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x_train.shape[1])]

            # Build 3 layer DNN with 10, 20, 10 units respectively.
            #shutil.rmtree("/tmp/iris_model")
            classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                  #hidden_units=[10, 20, 10],
                                                  #hidden_units=[8],
                                                  #hidden_units=[32],
                                                  #hidden_units=[128],
                                                  #hidden_units=[8,4],
                                                  #hidden_units=[32,16],
                                                  hidden_units=[128,64],
                                                  #dropout=0.5,
                                                  #hidden_units=[16,16,16,16],
                                                  #hidden_units=[32,32,32,32],
                                                  #Uniform exam
                                                  n_classes=2)
                                                  #model_dir="/tmp/iris_model")
            # Define the training inputs
            def get_train_inputs():
                x = tf.constant(x_train)
                y = tf.constant(t_train)

                return x, y

            # Fit model.
            #classifier.fit(input_fn=get_train_inputs, steps=2000)
            classifier.fit(x=x_train, y=t_train,batch_size=100,steps=4000)

            # Define the test inputs
            def get_test_inputs():
                x = tf.constant(x_test)
                y = tf.constant(t_test)

                return x, y

            # Evaluate accuracy.
            accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                               steps=1)["accuracy"]

            print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

            # Classify two new flower samples.
            def new_samples():
                return x_test

            predictions = list(classifier.predict(input_fn=new_samples))

            print(
              "New Samples, Class Predictions:    {}\n"
              .format(predictions))
            print(1-accuracy_score)
            Errors[i-1,trial-1]=1-accuracy_score
            sio.savemat('DNN_128_64_tabel1.mat', {'Errors':Errors})
            toc()
            #sio.savemat('DNN_128.mat', {'Errors':Errors})

if __name__ == "__main__":
    tic()
    main()
    toc()