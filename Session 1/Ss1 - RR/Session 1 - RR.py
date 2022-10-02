#!/usr/bin/env python
# coding: utf-8

# ## Khởi tạo dataframe

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv("x29.txt",header = None, index_col = 0)
df.head()


# In[2]:


df.info()


# In[3]:


df.describe()


# ## Split và chuẩn hóa

# In[4]:


# Số lượng
m = df.shape[0]

# Biến X
X = df.iloc[:,0:15]

# # Biến y
y = df.iloc[:,15]


# In[5]:


X.shape


# In[6]:


def normalize_and_add_ones(X):
    c = np.ones((60,15))
    for col_id in range(X.shape[1]): 
        for row_id  in range(X.shape[0]):
            c[row_id][col_id] = np.amax(X)[col_id+1]
    X_max = c
    
    c = np.ones((60,15))
    for col_id in range(X.shape[1]): 
        for row_id  in range(X.shape[0]):
            c[row_id][col_id] = np.amin(X)[col_id+1]
    X_min = c
    
    X_nor = (X - X_min) / (X_max - X_min)

    ones = np.array([[1] for _ in range(X_nor.shape[0])])
    return np.column_stack((ones, X_nor))

X = normalize_and_add_ones(X)
X


# In[7]:


X_train,y_train = X[:50],y[:50]
X_test,y_test = X[50:],y[50:]

print(y_train.shape)
print(X_train.shape)


# ## Xây dựng lớp RidgeRegression

# In[8]:


class RR:
    def __init__(self):
        return
    
    def fit(self, X_train, y_train, LAMBDA):...
        
    def predict(self, W, X_new):...
        
    def compute_RSS(self, y_new, y_predicted):...
        
    def get_the_best_LAMBDA(self, X_train, y_train):...
        
    def fit_grad(self, X_train, y_train, LAMBDA, learning_rate, max_num_epoch =100,batch_size = 128):...
        


# In[9]:


def fit(self, X_train, y_train, LAMBDA):
    assert len(X_train.shape) == 2 and X_train.shape[0] == y_train.shape[0]
    
    W = np.linalg.inv( X_train.transpose().dot(X_train) + LAMBDA * np.indentity(X_train.shape[1])
    ).dot(X_train.transpose()).dot(y_train)
    return W
    


# In[10]:


def predict(self, W, X_new):
    X_new = np.array(X_new)
    y_new = X_new.dot(W)
    return y_new
    


# In[11]:


def compute_RSS(self, y_new, y_predicted):
        loss = 1./ y_new.shape[0] * np.sum((y_new - y_predicted) ** 2)
        return loss


# In[12]:


def get_the_best_LAMBDA(self, X_train, y_train):
    def cross_validation(num_folds, LAMB):
        row_ids = np.array( range(X_train.shape[0]))
        valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
        valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
        train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
        aver_RSS = 0
        for i in range(num_folds):
            valid_part = {'X': X_train[valid_ids[i]], 'y': y_train[valid_ids[i]]}
            train_part = {'X': X_train[train_ids[i]], 'y': y_train[train_ids[i]]}
            W = self.fit_gradient_descent(train_part['X'], train_part['y'], LAMB)
            y_predicted = self.predict(W, valid_part['X'])
            aver_RSS += self.compute_RSS(valid_part['y'], y_predicted)
        return aver_RSS / num_folds
    
    def range_scan(best_LAMB, minimum_RSS, LAMB_values):
        for current_LAMB in LAMB_values:
            aver_RSS = cross_validation(num_folds = 5, LAMB = current_LAMB)
            if (aver_RSS < minimum_RSS):
                best_LAMB = current_LAMB
                minimum_RSS = aver_RSS
        return best_LAMB, minimum_RSS

    best_LAMB ,  minimum_RSS = range_scan( best_LAMB = 1, minimum_RSS=10000 ** 2, LAMB_values = range(50)) 

    # tiep tuc chia nho LAMB_values de tinh
    LAMB_values = [k*1./ 1000 for k in range( max(0,(best_LAMB - 1) * 1000), (best_LAMB + 1) * 1000 , 1)]
    best_LAMB , minimum_RSS = range_scan(best_LAMB = best_LAMB, minimum_RSS = minimum_RSS, LAMB_values = LAMB_values)

    return best_LAMB


# ## Model

# In[13]:


def fit_grad(self, X_train, y_train, LAMBDA, learning_rate, max_num_epoch =100,batch_size = 128):
    W = np.random.randn(X_train.shape[1])
    last_loss = 10e+8
    for ep in range(max_num_epoch):
        arr = np.array(range(X_train.shape[0]))
        np.random.shuffle(arr)
        X_train = X_train[arr]
        y_train = y_train[arr]
        total_minibatch = int(np.ceil(X_train.shape[0]/batch_size))
        for i in range(total_minibatch):
            index = i * batch_size
            X_train_sub = X_train[index:index+batch_size]
            y_train_sub = y_train[index:index+batch_size]
            grad = X_train_sub.T.dot(X_train_sub.dot(W)-y_train_sub)+ LAMBDA*W
            W = W - learning_rate*grad
            
        new_loss = self.compute_RSS(self.predict(W,X_train),y_train)
        if (np.abs(new_loss - last_loss) <= 1e-5):
            break
        last_loss = new_loss
        
        return W


# ## Main
# 

# In[14]:


ridgeRegression = RR()

best_LAMBDA = ridgeRegression.get_the_best_LAMBDA(X_train,y_train)
W_learned = ridgeRegression.fit(X_train = X_train, y_train = y_train, LAMBDA = 0.1)
y_predicted = ridgeRegression.predict( W = W_learned, X_new = X_test)
print ("Loss: ",ridgeRegression.compute_RSS(y_new = y_test, y_predicted = y_predicted))
print ("Best LAMBDA: ", best_LAMBDA)
print(X_train,X_test,y_train,y_test)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

