#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt


# # Dataset
# 
# Real estate agent table:

# |Area|Distance|Price|
# |---|---|---|
# |70|3|21200|
# |50|1|22010|
# |120|9|24305|
# |100|2|31500|
# 
# You can write the relationship with a 2-variable linear equation:
# 
# $
# \begin{equation}
# y = b + w_1.x_1 + w_2.x_2
# \end{equation}
# $
# 
# In a vector form:
# 
# $
# \begin{equation}
# y = b + (w_1 w_2).\binom{x_1}{x_2}
# \end{equation}
# $
# 
# Where
# $
# \begin{equation}
# W = (w_1 w_2)
# \end{equation}
# $
# and
# $
# \begin{equation}
# X = \binom{x_1}{x_2}
# \end{equation}
# $

# In[4]:


def generate_examples(num=1000):
    W = [1.0, -3.0]
    b = 1.0
    
    W = np.reshape(W, (2, 1))
    
    X = np.random.randn(num, 2)
    y = b + np.dot(X, W) + np.random.randn()
    
    y = np.reshape(y, (num, 1))
    
    return X, y


# In[5]:


X, y = generate_examples()


# In[6]:


print(X.shape, y.shape)


# In[7]:


print('X=', X[0], ' & y=', y[0])


# # Initialize Parameters

# The loss over **m** examples:
# 
# $
# \begin{equation}
# J = \frac{1}{2m} \sum_{i=1}^{m} (y - \hat{y})^2
# \end{equation}
# $
# 
# The objective of the gradient descent algorithm is to minimize this loss value.
# 
# Gradient Descent Objective is to 
# $
# \begin{equation}
# min(J)
# \end{equation}
# $

# In[8]:


class Model:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.randn(num_features, 1)
        self.b = np.random.randn()


# In[9]:


model = Model(2)
print('W=', model.W)
print('b=', model.b)


# # Forward Pass

# The gradient descent algorithm can be simplified in 4 steps:
# 
# 1. Get predictions y_hat for X with current values of W and b.
# 2. Compute the loss between y and y_hat
# 3. Find gradients of the loss with respect to parameters W and b
# 4. Update the values of W and b by subtracting the gradient values obtained in the previous step
# 
# Let's simplify our linear equation a bit more for an example:
# $
# \begin{equation}
# y = wx
# \end{equation}
# $
# 
# Let's plot J as a function of w
# 
# ![Loss vs Param](JvsW.png)
# 
# The gradients of loss with respect to w:
# 
# \begin{equation}
# \frac{dJ}{dw} = \frac{\delta{J}}{\delta{w}} = \lim_{\epsilon \to 0} \frac{J(w + \epsilon) - J(w)}{\epsilon}
# \end{equation}

# In[10]:


class Model(Model):
    def forward_pass(self, X):
        y = self.b + np.dot(X, self.W)
        return y


# In[11]:


preds = Model(2).forward_pass(np.random.randn(4, 2))
print(preds.shape)


# # Compute Loss
# 
# The loss over **m** examples:
# 
# $
# \begin{equation}
# J = \frac{1}{2m} \sum_{i=1}^{m} (y - \hat{y})^2
# \end{equation}
# $

# In[12]:


class Model(Model):
    def compute_loss(self, y, y_true):
        loss = np.sum(np.square(y - y_true))
        return loss/(2*y.shape[0])


# In[13]:


model = Model(2)

preds = model.forward_pass(X)
loss = model.compute_loss(y, preds)


# In[14]:


loss


# # Backward Pass
# 
# The gradient of loss with respect to bias can be calculated with:
# 
# $
# \begin{equation}
# \frac{dJ}{db} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y^{(i)}} - y^{(i)})
# \end{equation}
# $
# 
# $
# \begin{equation}
# \frac{dJ}{dW_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y^{(i)}} - y^{(i)}).x_j^{(i)}
# \end{equation}
# $

# In[15]:


class Model(Model):
    def backward_pass(self, X, y_true, y_hat):
        m = y_hat.shape[0]
        db = np.sum(y_hat - y_true)/m
        dW = np.sum(np.dot(np.transpose(y_hat - y_true), X), axis=0)/m
        return dW, db


# In[16]:


model = Model(2)
X, y = generate_examples()
y_hat = np.zeros(y.shape)

dW, db = model.backward_pass(X, y, y_hat)


# In[17]:


print(dW.shape, db.shape)


# # Update Parameters

# In[18]:


class Model(Model):
    def update_params(self, dW, db, lr):
        self.W = self.W - lr * np.reshape(dW, (self.num_features, 1))
        self.b = self.b - lr * db


# #  Training Loop

# In[19]:


class Model(Model):
    def train(self, x_train, y_train, iterations, lr):
        losses = []
        for i in range(iterations):
            y_hat = self.forward_pass(x_train)
            dW, db = self.backward_pass(x_train, y_train, y_hat)
            self.update_params(dW, db, lr)
            loss = self.compute_loss(y_hat, y_train)
            losses.append(loss)
            if i % 100 == 0:
                print('Iter: {}, Current loss: {:.4f}'.format(i, loss))
        return losses


# In[20]:


model = Model(2)


# In[21]:


X_train, y_train = generate_examples()


# In[22]:


losses = model.train(X_train, y_train, 1000, 3e-3)


# In[23]:


plt.plot(range(1000), losses);


# # Predictions

# In[24]:


model_untrained = Model(2)

X_test, y_test = generate_examples(500)
y_test = np.reshape(y_test, (y_test.shape[0], 1))

preds_untrained = model_untrained.forward_pass(X_test)
preds_trained = model.forward_pass(X_test)

plt.figure(figsize=(6, 6))
plt.plot(preds_untrained, y_test, 'rx')
plt.plot(preds_trained, y_test, 'bo')
plt.show()


# In[25]:


model.W


# In[26]:


model.b


# In[ ]:




