
# coding: utf-8

# In[3]:


import numpy as np
import sys
import csv 
train_input = sys.argv[1] 
validation_input = sys.argv[2]
train_out = sys.argv[3]
validation_out = sys.argv[4]
metrics_out = sys.argv[5]
num_epoch = int(sys.argv[6])
hidden_units = int(sys.argv[7])
init_flag = sys.argv[8]
learning_rate = float(sys.argv[9])

class read_file:
    def __init__(self, file_name):      
        open_file = csv.reader(open(file_name, "rb"), delimiter=",")
        file_list = list(open_file)
        x_matrix = np.matrix(file_list).astype("float")
        x_nobias = np.delete(x_matrix,0,1)  #x without bias
        self.x = np.insert(x_nobias,0,1.0,axis=1) #x with bias
        self.M = (np.prod(x_nobias.shape)/len(x_nobias))
        self.ycol = (np.asarray(x_matrix[:,0]).reshape(-1))
        self.K = (len(np.unique(self.ycol)))
        y_dg = np.diagflat(np.unique(self.ycol))
        self.y = np.identity(y_dg.shape[0])
        self.TE = len(x_nobias) 

#sigmoid function for z
def sigma(x):     
    z1= 1.0/(1.0+np.exp(-x))
    z=np.insert(z1,0,1.0,axis=0)
    return [z1,z]

#softmax function for y_cap
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def NNf(n,alpha, beta, x, y, yout):
    x1 = np.matrix(x)
    y1 = np.matrix(y)
    a= np.dot(alpha,(x1[n].T))
    z1,z = sigma(a)
    b= np.dot(beta,z)
    ycap = np.array(softmax(b)) 
    prediction = np.argmax(ycap)
    
    k = int(y[n])
    y_val = yout[k]
    loss = -float(np.dot(y_val,np.log(ycap)))
    return [loss,prediction,a,z1,z,b,ycap,y_val]

def NNb(n, alpha, beta, x, y, yout):
    x1 = np.matrix(x)
    y1 = np.matrix(y)   
    loss,prediction,a,z1,z,b,ycap,y_val = NNf(n, alpha, beta, x, y, yout)
    yout1 = np.matrix(y_val)
    
    ##dJ/dbeta
    gycap= -np.divide(yout1,(ycap.T))
    y_diag = np.diagflat(np.matrix(ycap)) 
    gb = np.dot(gycap,(y_diag - ycap*(ycap.T))) 
    gbeta = (gb.T)*(z.T) 
    
    ##dJ/dalpha
    beta1 = np.delete(beta,0,1) #removing 1st column of beta
    gz= (beta1.T)*(gb.T) 
    ga= np.multiply(np.multiply(gz,z1),1-z1) 
    galpha= ga*x1[n]
    return [galpha, gbeta]

def main():
    metricsout = ''
    train = read_file(train_input)
    validation = read_file(validation_input)
    
    alpha = np.asmatrix(np.random.uniform(-0.1,0.1,(hidden_units,train.M)))
    beta = np.asmatrix(np.random.uniform(-0.1,0.1,(train.K,hidden_units+1)))
    
    if init_flag == '1':
        alpha = np.asmatrix(np.random.uniform(-0.1,0.1,(hidden_units,train.M)))
        alpha = np.insert(alpha,0,0.0,axis=1)
        beta = np.asmatrix(np.random.uniform(-0.1,0.1,(train.K,hidden_units)))
        beta = np.insert(beta,0,0.0,axis=1)
    elif init_flag == '2':
        alpha = np.zeros((hidden_units,train.M+1))
        beta = np.zeros((train.K,hidden_units+1))
    
    for i in range(num_epoch): 
        L_t = []
        L_v = []
        Pred_v=[]
        #updating alpha and beta for all the training examples
        for j in range(0,train.TE):
            galphat, gbetat = NNb(j, alpha, beta, train.x, train.ycol, train.y)
            alpha = alpha - learning_rate*galphat #updating alpha
            beta = beta - learning_rate*gbetat #updating beta  
           
        for j in range(0,train.TE):
            loss_t,predictiont,a,z1,z,b,ycap,y_val= NNf(j, alpha, beta, train.x, train.ycol, train.y)
            L_t.append(loss_t)

        Loss_train= np.sum(L_t)/train.TE

        for j in range(0,validation.TE):
            loss_v,prediction,a,z1,z,b,ycap,y_val = NNf(j, alpha, beta, validation.x, validation.ycol, validation.y)
            L_v.append(loss_v)
            
        L_validation= np.sum(L_v)/validation.TE
        
        metricsout += 'epoch={} crossentropy(train): {:.11f}\n'.format(i+1, Loss_train)
        metricsout += 'epoch={} crossentropy(validation): {:.11f}\n'.format(i+1, L_validation)
    
    #prdiction
    t_label = []
    for j in range(0,train.TE):
        loss,prediction_t,a,z1,z,b,ycap,y_val= NNf(j, alpha, beta, train.x, train.ycol, train.y)
        t_label.append(prediction_t)
    
    v_label = []
    for j in range(0,validation.TE):
        loss,prediction_v,a,z1,z,b,ycap,y_val = NNf(j, alpha, beta, validation.x, validation.ycol, validation.y)
        v_label.append(prediction_v)

    #error
    training_err = 1 - float(sum(t_label==train.ycol))/len(train.ycol)
    metricsout += 'error(train): {:.2f}\n'.format(training_err)
    validation_err = 1 - float(sum(v_label==validation.ycol))/len(validation.ycol)
    metricsout += 'error(validation): {:.2f}\n'.format(validation_err)

    train_labels_print = ''
    for i in range(len(t_label)):
        train_labels_print = train_labels_print + str(t_label[i]) + '\n'    

    val_labels_print = ''
    for i in range(len(v_label)):
        val_labels_print = val_labels_print + str(v_label[i]) + '\n'   
     
    with open(metrics_out, 'w') as f:
        f.write(metricsout)
    f.closed
    with open(train_out, 'w') as f:
        f.write(train_labels_print)
    f.closed
    with open(validation_out, 'w') as f:
        f.write(val_labels_print)
    f.closed
            
        
if __name__ == "__main__":
    main()


