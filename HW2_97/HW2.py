#!/usr/bin/env python
# coding: utf-8

# # HOME WORK 2 (HW2_97)
# 
# ## MOHAMMAD SADEGH GHASEMI (965051511)

# In[1]:


import glob
import os
import h5py
import imageio
import string
import numpy as np
import SimpSOM as sps
import matplotlib.pyplot as plt
import scipy.io

from sklearn import svm
from sklearn.model_selection import train_test_split
from PIL import Image
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from minisom import MiniSom
from sklearn import datasets
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# <hr><hr>
# 
# ## 1.Mushroom Recognition

# SVM from scikit-learn is used for this problem. Its implementation is based on libsvm.

# In[2]:


# LOAD DATA MATRIX
mat = scipy.io.loadmat('mushroom_pre.mat')
y = mat['Y'].flatten()
X = mat['X'].T


# In[3]:


# SPLIT TRAIN/TEST DATA (40% TEST, 60% TRAIN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[4]:


# LINEAR SVM WITH C=100
# THE LARGER C GETS, THE MORE THE MODEL GOES TO HARD MARGIN.(THE BIGGER THE C, THE LESS ERROR)
clf = svm.SVC(kernel='linear', C=100)


# ### Penalty parameter C of the error term<br>
# A large value of C basically tells the model that we do not have that much faith in our data’s distribution, and will only consider points close to line of separation.
# 
# A small value of C includes more/all the observations, allowing the margins to be calculated using all the data in the area.
# 
# این متغییر اهمیت خطا را مشخص می کند و هرچه مقدار بیشتری داشته باشد، پذیرش خطا کمتر می شود و در نتیجه مدل به سمت هارد مارجین میل می کند.<br>
# بنابر این با درنظر گرفتن مقدار ۱۰۰ برای این متغییر، در واقع مدل تقریبا هارد مارجین می شود، حاشیه کوچکتر درنظر  گرفته می شود، نزدیک ترین داده ها به مرز جدایی ساز به عنوان بردارهای پشتیبان انتخاب می شوند و به این ترتیب در فضای حاشیه، داده ای نخواهیم داشت و خطای حاشیه ای کاهش می یابد.

# In[5]:


# TRAIN MODEL
clf.fit(X_train, y_train)


# In[6]:


# PRINT ACCURACY
accuracy = clf.score(X_test, y_test)
print(
    f"""
    C: {clf.C}
    ACCURACY: {round(accuracy, 3) * 100}%""")


# In[7]:


# SET C TO 0.0001
# THE SMALLER C GETS, THE MORE THE MODEL GOES TO SOFT MARGIN.(THE SMALLER THE C, THE BIGGER THE MORGIN, THE MORE ERROR)
clf.C = 0.0001

# TRAIN MODEL
clf.fit(X_train, y_train) 


# In[8]:


# PRINT ACCURACY
accuracy = clf.score(X_test, y_test)
print(
    f"""
    C: {clf.C}
    ACCURACY: {round(accuracy, 2) * 100}%""")


# In[9]:


# COMPARE ACCURACY FOR DIFFERENT VALUES OF C
c_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
for c in c_list:
    clf.C = c
    clf.fit(X_train, y_train)
    print(f"C: {clf.C}")
    print(f"ACCURACY: {round(clf.score(X_test, y_test), 3) * 100}%")


# <hr>
# <hr>
# 
# ## 2. Approximate the bellow function in range [-3, 3] using Radial Basis Function Network
# 
# <br>
# 
# ## f(x) = 0.5 + cos(0.8π x)

# In[10]:


# DEFINE RADIAL BASIS FUNCTION (RBF NET NEURONS)
def rbf(x, c, s):
    """Radial Basis Function.
    Args:
        x: data point (scalar)
        c: x's cluster center
        s: standard deviation

    Returns:
        exp(-((x-c)^2) / (2 * s^2))
    """
    return np.exp(-1 / (2 * s**2) * (x-c)**2)


# In[11]:


# KMEANS CLUSTERING FOR CHOOSING BEST CENTERS OF THE DATA
def kmeans(X, k):
    """Performs k-means clustering for 1D input
    
    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters
    
    Returns:
        ndarray -- A kx1 array of final cluster centers
    """
 
    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False
 
    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
 
        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)
 
        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)
 
        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()
 
    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)
 
    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])
 
    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))
 
    return clusters, stds


# ## RBFNet
# #### این شبکه طوری طراحی شده است که می تواند مراکز را به صورت رندم از بین داده های آموزشی انتخاب کند، یا با استفاده از روش خوشه بندی، بهترین مراکز را پیدا کند.
# #### به صورت پیشفرض مراکز با استفاده از الگوریتم خوشه بندی انتخاب می شوند.
# #### اگر در هنگام فراخوانی استفاده از خوشه بندی غیر فعال شود، مراکز از بین داده های آموزشی انتخاب می شوند و اگر تعداد مراکز کمتر از تعداد داده های آموزشی باشد، به صورت تصادفی از بین داده های آموزشی انتخاب خواهند شد.
# #### در صورت استفاده از خوشه بندی، انحراف معیار برای هر خوشه به صورت مجزا و با توجه به پراکندگی هر خوشه، تنظیم می شود.
# #### این امکان وجود دارد که به هنگام ایجاد مدل، انحراف معیار به صورت عددی ثابت تنظیم شود.
# #### اگر انحراف معیار تنظیم نشود و از خوشه بندی نیز استفاده نشود، انحراف معیار با فرمول زیر محاسبه می شود:
# #### اندازه ی بیشترین فاصله بین مراکز، تقسیم بر ریشه ی دومِ دوبرابر تعداد مراکز
# #### || Ci - Cj || / sqrt(2*k)

# In[12]:


# RADIAL BASIS FUNCTION NETWORK
class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=None, center_selection_method="random", lr=0.01, epochs=300, rbf=rbf, inferStds=True, std=None):
        self.stds = None
        self.k = k or X.size
        self.center_selection_method = center_selection_method.lower()
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        self.std = std
 
        self.w = np.random.randn(self.k)
        self.b = np.random.randn(1)

    def fit(self, X, y, use_kmeans=True):
        if use_kmeans:
            if self.inferStds:
                # compute stds from data
                self.centers, self.stds = kmeans(X, self.k)
            else:
                # use a fixed std 
                self.centers, _ = kmeans(X, self.k)
                if self.stds is None:
                    if self.std is not None:
                        self.stds = np.repeat(self.std, self.k)
                    else:
                        dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
                        self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)
        else:
            if self.center_selection_method == "random":
                self.centers = np.random.choice(X, self.k)
            elif self.center_selection_method == "first":
                self.centers = X[:self.k]
            elif self.center_selection_method == "last":
                self.centers = X[self.k:]
            else:
                raise ValueError('invalid center_selection_method!')
                
            if self.std:
                self.stds = np.repeat(self.std, self.k)
            else:
                dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
                self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2

                # backward pass
                error = -(y[i] - F).flatten()

                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
                             
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)


# In[13]:


# FUNCTION TO APPROXIMATE
def fx(X):
    """
    f(x) = 0.5 + cos(0.8π x)
    
    Returns:
        0.5 + np.cos(0.8 * np.pi * X)
    """
    return 0.5 + np.cos(0.8 * np.pi * X)


def plot_fx_vs_rbf_approximation(X, y, rbf_net_model):
    """Plots fx vs it's RBF apptoximation.
    
    Args:
        X: 1D array of input data
        y: desired output
        rbf_net_model: trained RBFNet object
    """
    # PREDICT FUNCTION OUTPUTS
    y_pred = rbf_net_model.predict(X)

    # PLOT FX VS RBF FX-APPROXIMATION
    plt.plot(X, y, '-o', label='true')
    plt.plot(X, y_pred, '-o', label='RBF-Net')
    plt.legend()

    plt.tight_layout()


# In[14]:


# DEFINE INPUT AND OUTPUT DATA
X = np.linspace(-3, 3, 100)
y = fx(X)


# ## الف: انحراف معیار ۱ در نظر گرفته شود

# k (number of hidden units) = input size<br>
# std (spread) = 1<br>
# تعداد مراکز به اندازه ی داده های آموزشی و انحراف معیار ۱ در نظر گرفته شده است

# In[15]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(std=1)
rbfnet.fit(X, y, use_kmeans=False)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# ## ب-1: تعداد واحدهای مخفی نصف تعداد داده های آموزشی در نظر گرفته شود

# k (number of centers) = input_size / 2<br>
# std (spread) = 1<br>
# تعداد مراکز به اندازه ی نصف داده های آموزشی و انحراف معیار ۱ در نظر گرفته شده است
# ### مراکز به صورت تصادف انتخاب شده اند

# In[16]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(k=X.size//2, std=1)
rbfnet.fit(X, y, use_kmeans=False)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# ## ب-2: تعداد واحدهای مخفی یک سوم تعداد داده های آموزشی در نظر گرفته شود

# k (number of centers) = input_size / 3<br>
# std (spread) = 1<br>
# تعداد مراکز به اندازه ی یک سوم داده های آموزشی و انحراف معیار ۱ در نظر گرفته شده است
# ### مراکز به صورت تصادف انتخاب شده اند

# In[17]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(k=X.size//3, std=1)
rbfnet.fit(X, y, use_kmeans=False)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# ## تعداد واحدهای مخفی یک بیستم تعداد داده های آموزشی در نظر گرفته شده است
# k (number of centers) = input_size / 20<br>
# std (spread) = 1<br>
# تعداد مراکز به اندازه ی یک بیستم داده های آموزشی و انحراف معیار ۱ در نظر گرفته شده است
# ### مراکز به صورت تصادف انتخاب شده است

# In[18]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(k=X.size//20, std=1)
rbfnet.fit(X, y, use_kmeans=False)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# ### این مثال نشان می دهد که هرچه تعداد مراکز کمتر شود دقت مدل کمتر می شود، چرا که با کاهش تعداد، فاصله ی بین مراکز بیشتر می شود و در نتیجه تابع اصلی به اندازه ی حالتی که تعداد مراکز با تعداد داده های آموزشی یکسان در نظر گرفته شود نمی تواند پوشش داده شود.<br>
# ###  اگر مراکز با توزیع مناسبی انتخاب شود، تقریب مناسب تری از تابع هدف، به دست می آید، این مثال را با انتخاب مراکز از داده های اول (انتخاب نا متوازن) نشان می دهیم.

# ### در این مثال تعداد مراکز را یک سوم داده های ورودی در نظر گرفته و آن ها را از داده های ابتدایی انتخاب می کرده ایم. مشاهده می شود که تابع در بازه ی ابتدایی بهتر تقریب زده شده است چرا که مراکز از این بازه انتخاب شده اند.
# 

# In[19]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(k=X.size//3, std=1, center_selection_method="first")
rbfnet.fit(X, y, use_kmeans=False)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# ### در این مثال تعداد مراکز را یک پنجم تعداد داده های ورودی در نظر گرفته و آن ها را از طریق خوشه بندی انتخاب کرده ایم. انحراف معیار نیز ۰.۵ در نظر گرفته شده است. مشاهده می شود که با انتخاب مراکز مناسب، تابع به خوبی تقریب زده شده است

# In[20]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(k=X.size//5, std=.5, inferStds=False)
rbfnet.fit(X, y, use_kmeans=True)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# ## ج: حالت های الف و ب را با انحراف معیار ۰.۲ و ۲ تکرار کنید و پاسخ ها را تحلیل کنید

# ### با توجه به اینکه فاصله ی بین داده های ورودی کوچک است (چون ۱۰۰ داده بین ۳- و ۳+ انتخاب شده است) انحراف معیار ۰.۲ برای تقریب، مقدار کوچکی نیست، اما به طور کلی اگر انحراف معیار کوچک انتخاب شود به علت اینکه همسایگی تنها در فاصله ی کوچکی تا مراکز در نظر گرفته می شود، تابع به صورت نویزی تقریب زده می شود و اگر انحراف معیار بزرگ در نظر گرفته شود تابع با تغییر کم و به صورت میان گین نرم تقریب زده می شود.

# تعداد واحدهای مخفی برابر با تعداد داده های آموزشی و انحراف معیار برابر با 0.2 در نظر گرفته شده است
# <br>
# k (number of centers) = input_size <br>
# std (spread) = 0.2<br>

# In[22]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(std=.2)
rbfnet.fit(X, y, use_kmeans=False)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# تعداد واحدهای مخفی برابر با تعداد داده های آموزشی و انحراف معیار برابر با 0.05 در نظر گرفته شده است
# <br>
# k (number of centers) = input_size <br>
# std (spread) = 0.05<br>
# ### با توجه به کوچک بودن انحراف معیار، تابع با نویز تقریب زده می شود

# In[23]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(std=.05)
rbfnet.fit(X, y, use_kmeans=False)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# تعداد واحدهای مخفی برابر با تعداد داده های آموزشی و انحراف معیار برابر با 2 در نظر گرفته شده است
# <br>
# k (number of centers) = input_size <br>
# std (spread) = 0.05<br>
# ### با توجه به بزرگ بودن انحراف معیار و شرکت تعداد زیادی از مراکز برای تقریب تابع در هر نقطه، مدل تابع هدف را به صورت میانگین خیلی نرم و با تغییرات کم، تقریب می زند

# In[24]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(std=2)
rbfnet.fit(X, y, use_kmeans=False)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# تعداد واحدهای مخفی برابر با یک سوم تعداد داده های آموزشی و انحراف معیار برابر با 2 در نظر گرفته شده است
# <br>
# k (number of centers) = input_size//3 <br>
# std (spread) = 0.05<br>
# ### انحراف معیار همچنان مقدار بزرگی است(چون فاصله ی داده های ورودی بسیار کوچک در نظر گرفته شده است) اما با کاهش تعداد مراکز، تقریب، اندکی بهتر می شود

# In[25]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(std=2, k=X.size//3, inferStds=False)
rbfnet.fit(X, y, use_kmeans=True)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# تعداد واحدهای مخفی برابر با یک سوم تعداد داده های آموزشی و انحراف معیار برابر با 0.05 در نظر گرفته شده است
# <br>
# k (number of centers) = input_size//3 <br>
# std (spread) = 0.05<br>
# ### انحراف معیار کوچک انتخاب شده است و با توجه به کاهش تعداد واحدهای مخفی به یک سوم تعداد داده های آموزشی، نویز تقریب تابع بیش از حالتی می شود که تعداد واحدها به اندازه ی تعداد داده های آموزشی است

# In[26]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(std=0.05, k=X.size//3, inferStds=False)
rbfnet.fit(X, y, use_kmeans=True)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# ## د: یک نویز گوسی با متوسط صفر ایجاد کرده و با مقدار تابع اصلی جمع کرده و چگونگی عملکرد شبکه ها را در مقایسه با حالت بدون نویز بررسی کنید. نسبت خطا به توان نویز را نیز اندازه بگیرید

# ###  یک نویز گوسی با میانگین ۰ و انحراف معیار استاندارد ۰.۲ به داده ها اعمال شده است.

# In[27]:


# DEFINE INPUT AND OUTPUT DATA
X = np.linspace(-3, 3, 100)
y = fx(X)

noise = np.random.normal(0, .2, size=100)
noisy_y = fx(X) + noise


# ### در این مثال تعداد واحدهای مخفی برابر با یک پنجم تعداد داده های ورودی و انحراف معیار برابر با ۰.۵ در نظر گرفته شده است. برای انتخاب مراکز از روش خوشه بندی استفاده شده است. مشاهده می شود که تقریب نرم تری از  تابع نویزی اصلی زده شده است.<br>مجموع خطای ۱۰۰ داده ی ورودی ۱۶.۶۴ شده است.
# 

# In[28]:


# TRAIN RBF NETWORK
noisy_rbfnet = RBFNet(k=X.size//5, std=.5, inferStds=False)
noisy_rbfnet.fit(X, noisy_y, use_kmeans=True)
plot_fx_vs_rbf_approximation(X, noisy_y, noisy_rbfnet)


# In[29]:


noisy_errors = np.abs((noisy_rbfnet.predict(X).flatten() - noisy_y))
# noisy_sse = np.sum(np.power(noisy_errors, 2))
noisy_average_error = np.average(noisy_errors)
print("Sum of errors (sum(|OUTPUT-noisy_y|):\n"
      f"    {np.sum(noisy_errors)}\n")
print("Average error:\n"
      f"    {noisy_average_error}\n")
print("Sum of <ERRORS raised to the NOISE>:\n"
      f"    {np.sum(np.power(noisy_errors, noise))}\n"
      "    (خطا به ازای هر ورودی به توان نویز مربوط به همان ورودی رسیده است و سپس مجموع گرفته شده است)")
print("\n<Sum of errors> raised to the <Sum of noises>:\n"
      f"    {np.power(np.sum(noisy_errors), np.sum(noise))}\n"
      "    (مجموع خطا به توان مجموع نویز)")


# ### در این مثال تعداد واحدهای مخفی برابر با یک پنجم تعداد داده های ورودی و انحراف معیار برابر با ۰.۵ در نظر گرفته شده است. برای انتخاب مراکز از روش خوشه بندی استفاده شده است. مشاهده می شود که تقریب نرم تری از  تابع نویزی اصلی زده شده است.
# 

# In[30]:


# TRAIN RBF NETWORK
rbfnet = RBFNet(k=X.size//5, std=.5, inferStds=False)
rbfnet.fit(X, y, use_kmeans=True)
plot_fx_vs_rbf_approximation(X, y, rbfnet)


# In[31]:


errors = np.abs((rbfnet.predict(X).flatten() - y))
# noisy_sse = np.sum(np.power(errors, 2))
average_error = np.average(errors)
print("Sum of errors (sum(|OUTPUT-y|):\n"
      f"    {np.sum(errors)}\n")
print("Average error:\n"
      f"    {average_error}")


# In[32]:


print("Errors ratio:\n"
      f"    {np.sum(noisy_errors) / np.sum(errors)}\n"
      "    (نسبت خطا با وجود نویز به خطا بدون نویز)")


# <hr><hr>
# 
# ## چگونه می توان از توابع پایه شعاعی برای اینترپولیشن تصاویر استفاده کرد؟

# ### از توابع پایه ی شعاعی می توان برای دیفرمیشن تصاویر و همچنین بازسازی تصاویر آسیب دیده استفاده کرد.
# ### در مثال زیر از تصویر پیکسل ها را به صورت رندم انتخاب می کنیم و سپس تصویر اولیه را بازسازی می کنیم.

# In[33]:


montage = Image.open("montage.gif").convert('L')
np_montage = np.asanyarray(montage)


# In[34]:


plt.imshow(montage);


# ### Given a random-sampled selection of pixels from an image, scipy.interpolate.griddata could be used to interpolate back to a representation of the original image.

# In[35]:


def make_interpolated_image(nsamples, img):
    """Make an interpolated image from a random selection of pixels.

    Take nsamples random pixels from img and reconstruct the image using
    scipy.interpolate.griddata.

    """

    ix = np.random.randint(img.shape[1], size=nsamples)
    iy = np.random.randint(img.shape[0], size=nsamples)
    samples = img[iy,ix]
    int_im = griddata((iy, ix), samples, (Y, X))
    return int_im


# A meshgrid of pixel coordinates
ny, nx = np_montage.shape
X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))

# Create a figure of nrows x ncols subplots, and orient it appropriately
# for the aspect ratio of the image.
nrows, ncols = 2, 2
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6,4), dpi=100)
if nx < ny:
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.set_figwidth(h), fig.set_figheight(w)

# Convert an integer i to coordinates in the ax array
get_indices = lambda i: (i // nrows, i % ncols)

# Sample 100, 1,000, 10,000 and 100,000 points and plot the interpolated
# images in the figure
for i in range(4):
    nsamples = 10**(i+2)
    axes = ax[get_indices(i)]
    axes.imshow(make_interpolated_image(nsamples, np_montage),
                          cmap=plt.get_cmap('Greys_r'))
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title('nsamples = {0:d}'.format(nsamples))
filestem = os.path.splitext(os.path.basename("montage_interpolated"))[0]
plt.savefig('{0:s}_interp.png'.format(filestem), dpi=100)


# <hr><hr>
# 
# ## 4- شبکه ی رقابتی

# In[36]:


# LOAD DATA MATRIX
alphabet_matrix = scipy.io.loadmat('alphachars.mat')
alphabet = alphabet_matrix['alphabet'].T


# In[37]:


def plot_alphabet(alphabet_data):
    """Plots alphabetic character images.
    """
    for i in range(alphabet_data.shape[0]):
        plt.subplot(alphabet_data.shape[0]//6 + 1, 6, i + 1)
        a = alphabet_data[i, :].reshape((7, 5))
        plt.imshow(a)


def plot_misclassified_vs_true_alphabet(X_error, predict, y, prototypes, n=None):
    """Plots input, (wrong) predicted character and true character. 
    """
    print("  input          predicted    true")
    if n is None:
        n_row = len(X_error)
    else:
        n_row = n
        
    n_col = 3
    for i in range(n_row):
        plt.subplot(n_row, n_col, i + 1 + i * 2)
        a = X_error[i].reshape((7, 5))
        plt.imshow(a)
        
        plt.subplot(n_row, n_col, i + 2 + i * 2)
        b = prototypes[predict[i].argmax()].reshape((7, 5))
        plt.imshow(b)
        
        plt.subplot(n_row, n_col, i + 3 + i * 2)
        c = prototypes[y[i].argmax()].reshape((7, 5))
        plt.imshow(c)


def normalize_input(X, prototype_norm):
    """Normalizes the size of X with respect to the prototype_norm.
    """
    Y = X.astype(np.float)
    for i in range(Y.shape[0]):
        Y[i, :] = Y[i, :] / np.linalg.norm(Y[i, :]) * prototype_norm
    return Y


# In[38]:


# DEFINE LAYER 1 NORMALIZED PROTOTYPE WEIGHTS AND BIAS
# ALPHABETS THEMSELVES ARE PROTOTYPES
W1 = alphabet.astype(np.float) 
S, R = W1.shape 
B = np.ones((S,)) * R 

# PROTOTYPES NEED TO BE WITH THE SAME NORM SO WE NORMALIZE 
# THEM WITH RESPECT TO SMALLEST ONE OF THEM
norm_W1 = np.linalg.norm(alphabet, axis=1) 
min_norm_idx = np.argmin(norm_W1) 
prototype_norm = norm_W1[min_norm_idx]  
for i in range(min_norm_idx): 
    W1[i, :] = W1[i, :] / norm_W1[i] * prototype_norm 
for i in range(min_norm_idx + 1, norm_W1.size): 
    W1[i, :] = W1[i, :] / norm_W1[i] * prototype_norm 

# DEFINE LAYER 2 WEIGHT MATRIX
# epsilon SHOULD BE SMALLER THAN 1/(S-1)
# SO WE RANDOMLY CHOOSE IT TO BE BETWEEN 0.01 AND (1/(S-1))
epsilon = 0;
while epsilon < 0.01:
    epsilon = np.random.rand() % (1/(S-1))

W2 = np.ones((S, S), dtype=np.float) * -epsilon
np.fill_diagonal(W2, 1)

# INPUT DATA ARE CREATED FROM THE ORIGINAL ALPHABET WITH
# SOME NOISE. 4 SAMPLE OF ALPHABET ARE CREATED WITH 4
# DIFFERENT AMOUNTS OF NOISES
P1 = alphabet + np.abs(np.random.randn(S, R) * 0.3)
P2 = alphabet + np.abs(np.random.randn(S, R) * 0.4)
P3 = alphabet + np.abs(np.random.randn(S, R) * 0.5)
P4 = alphabet + np.abs(np.random.randn(S, R) * 0.8)

X_train = np.vstack((P1, P2, P3, P4))

# THE OUTPUT OF EACH CHARACTER(ALPHABETIC DATAPOINT) IS ONE HOT OF ITS INDEX
# IN ENGLISH ALPHABET ORDER 
train_size = X_train.shape[0]
Y_train = np.zeros((train_size, S))
for i in range(train_size):
    Y_train[i, i % S] = 1

P4 = alphabet + np.abs(np.random.randn(S, R) * 0.8)


# In[39]:


print("Prototypes")
plot_alphabet(alphabet)


# In[40]:


print("Sample one (with the least noise)")
plot_alphabet(P1)


# In[41]:


print("Sample 4 (with the highest noise)")
plot_alphabet(P4)


# ### این شبکه مطابق آنچه در کلاس تدریس شده،  پیاده سازی شده است
# #### داده های ورودی باید هم اندازه ی بردارهای پروتوتایپ باشند

# In[42]:


X_train_norm = normalize_input(X_train, prototype_norm)


# In[43]:


print(f"norm 2 of the first normalized input: {np.linalg.norm(X_train_norm[0])}")
print(f"norm 2 of the first prototype: {np.linalg.norm(W1[0])}")


# ### لایه ی اول شباهت بین داده ی ورودی با بردارهای الگو را اندازه گیری می کند

# In[44]:


# LAYER 1
def a_1(x):
    return W1.dot(x) + B


# ### لایه ی دوم، لایه ی رقابتی است و خروجی نهایی آن وکتوری وان هات است که نشان دهنده ی کلاس پیش بینی شده است

# In[45]:


# LAYER 2
def a_2(a1):
    a2 = a1.copy()
    a2_new = W2.dot(a2)
    # in fact (for this problem) we can simply select maximum as winner
    # but we will do the method introduced in class (with max_iter 50)
    max_iter = 50
    
    j = 1
    while not np.allclose(a2, a2_new) and j < max_iter:  #  or not np.allclose(a2_new, a2)
        a2[:] = a2_new[:]
        a2_new = W2.dot(a2)
        j += 1
    
    output = np.zeros_like(a2_new)
    output[a2_new.argmax()] = 1
    
    return output


def compet(x):
    """Predicts the class of each input using trained competitive network
    """
    return a_2(a_1(x))


# In[46]:


def predict_and_check(X, Y, compet_net):
    n = X.shape[0]
    errors = []
    predicted_output = []
    desired_output = []

    for i in range(n):
        x = X[i]
        y = compet_net(x)
        if np.argmax(y) != np.argmax(Y[i]):
            errors.append(x)
            predicted_output.append(y)
            desired_output.append(Y_train[i])
    return errors, predicted_output, desired_output


# In[47]:


errors, predicted_output, desired_output = predict_and_check(X_train_norm, Y_train, compet)
plot_misclassified_vs_true_alphabet(errors, predicted_output, desired_output, alphabet)


# In[48]:


print(f"""Total errors: {len(errors)}
Accuracy: {100. - len(errors) / X_train_norm.shape[0]}""")


# 
# ### مشاهده می شود که دقت مدل روی داده های آموزشی تقریبا برابر با ۱ است
# یک مجموعه داده ی جدید به عنوان داده های تستی ایجاد می کنیم و مدل را با آن می سنجیم

# In[53]:


# TEST DATA
X_test = alphabet + np.abs(np.random.randn(S, R) * np.random.randint(30, 90, R)/ 100)
X_test_norm = normalize_input(X_test, prototype_norm)

# THE OUTPUT OF EACH CHARACTER(ALPHABETIC DATAPOINT) IS ONE HOT OF ITS INDEX
# IN ENGLISH ALPHABET ORDER 
test_size = X_test.shape[0]
Y_test = np.zeros((test_size, S))
for i in range(test_size):
    Y_test[i, i % S] = 1


# In[54]:


errors, predicted_output, desired_output = predict_and_check(X_test_norm, Y_test, compet)
plot_misclassified_vs_true_alphabet(errors, predicted_output, desired_output, alphabet)


# ### مشاهده شد که دقت مدل بر روی داده های تستی نیز تقریبا ۱ است. بنابراین نیازی به آموزش لایه ی اول وجود ندارد و الگوهای هر کلاس به خوبی انتخاب شده اند

# ## USE SOM for this Problem

# In[55]:


# ALPHABET FOR MAP
labels = [a for a in string.ascii_letters.split('z')[1][:]]


# In[57]:


#Build a network 16x16 with a weights format taken from the X_train_norm and activate Periodic Boundary Conditions. 
net = sps.somNet(16, 16, X_train_norm, PBC=True)

#Train the network for 10000 epochs and with initial learning rate of 0.1. 
net.train(0.01, 20000)

#Save the weights to file
net.save('SimpSOM_16_by_16_Weights')


# In[58]:


_ = net.project(X_train_norm, labels=labels, show=True, printout=False)


# ### در نقشه ی تولید شده حروفی که از نظر نوشتاری شبیه به هم هستند، نزدیک به هم قرار گرفته اند
# [I-T-J]-[G-C-O-Q-D]-[E-F]-[P-R]-[M-N-H]...
# 
# به همین ترتیب می توان نقشه های با ابعاد متفاوت نیز ایجاد کرد.کافی است تعداد نرون های مشبک
# را در هنگام ایجاد مدل مشخص کنیم

# <hr><hr>
# 
# ## 5- Use SOM network for digits dataset

# In[59]:


# LOAD DATA MATRIX
digit_mat = h5py.File('digits.mat', 'r')
digits = np.array(digit_mat['data'])


# ### 30 PERCENT OF DATA FOR TRAIN

# In[60]:


X_train = digits[:, ::3, :]


# In[61]:


X_train.shape


# #### برای هر رقم ۳۶۷ نمونه انتخاب شده است که هر کدام دارای ابعاد ۱ * ۲۵۶ و درواقع ۱۶ * ۱۶ هستند

# In[62]:


# merge 2 first dimentions
X_train_flat = X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2])
# each 367 sample is for one digit and digits are in this order: [1,2,3,4,5,6,7,8,9,0]
Y_train = (np.arange(X_train_flat.shape[0])//367 + 1) % 10


# #### با توجه به اینکه ارقام ترتیب تصادفی ندارند، بهتر است ترتیب داده های آموزشی را به صورت تصادف تغییر دهیم

# In[63]:


# shuffle the data
idx = np.arange(X_train_flat.shape[0])
np.random.shuffle(idx)
X_train_flat = X_train_flat[idx]
Y_train = Y_train[idx]


# ### نمایش یک رقم به تصادف

# In[64]:


plt.imshow(X_train_flat[np.random.randint(1, 3670, 1),:].reshape(16, 16).T);


# ### SOM training (using MiniSom)
# آموزش مدل با نرخ یادگیری ۰.۲ و انحراف معیار 3 (برای همسایگی)

# In[65]:


som = MiniSom(30, 30, 256, sigma=3,
              learning_rate=0.2, neighborhood_function='triangle')
som.pca_weights_init(X_train_flat)
print("Training...")
som.train_random(X_train_flat, 20000)  # random training
print("\n...ready!")


# In[66]:


plt.figure(figsize=(8, 8))
wmap = {}
im = 0
for x, t in zip(X_train_flat, Y_train):  # scatterplot
    w = som.winner(x)
    wmap[w] = im
    plt. text(w[0]+.5,  w[1]+.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
plt.savefig('som_digts.png')


# <hr><hr>
# 
# ## 6- Use keras and CNN for classifing Airplanes vs no-Airplanes

# In[67]:


# path to Airplane pictures
planes_path = './planes/'
# path to no-Airplane pictures
no_planes_path = './no-planes'

planes_file_paths = glob.glob(os.path.join(planes_path, '*.png'))
no_planes_file_paths = glob.glob(os.path.join(no_planes_path, '*.png'))


# In[68]:


def shuffle(X, Y=None):
    """Shuffles the first dimention of X (Y).
    
    Args:
        X: numpy nd-array
        Y: numpy nd-array
    """
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y must be with the same size!')

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    if Y is not None:
        Y = Y[idx]
    
    return X if Y is None else (X, Y)


# در این بخش تصاویر مربوط به هواپیما و غیر هواپیما از مسیر مربوطه خوانده می شود. ۱۵۰۰ داده از هرکدام برای آموزش، ۵۰۰ داده برای تست و ۵۰۰ داده برای اعتبار سنجی استفاده می شود. داده ها قبل از استفاده برای آموزش و تست به صورت تصادفی با هم ادغام می شوند. برای نرمال سازی تمام تصاویر تقسیم بر ۲۵۵ شده اند تا در بازه ی ۰ و ۱ قرار بگیرند

# In[69]:


plane_images = [imageio.imread(path) for path in planes_file_paths]
plane_images = shuffle(np.asarray(plane_images))
plane_images = plane_images / 255

no_plane_images = [imageio.imread(path) for path in no_planes_file_paths]
no_plane_images = shuffle(np.asarray(no_plane_images))
no_plane_images = no_plane_images / 255

train_size = 1500
test_size = 500
val_size = 500

X_plane_train = plane_images[:train_size]
X_plane_test = plane_images[train_size:(train_size + test_size)]
X_plane_val = plane_images[(train_size + test_size):(train_size + test_size + val_size)]
Y_plane_train = np.ones(X_plane_train.shape[0])
Y_plane_test = np.ones(X_plane_test.shape[0])
Y_plane_val = np.ones(X_plane_val.shape[0])

X_no_plane_train = no_plane_images[:train_size]
X_no_plane_test = no_plane_images[train_size:(train_size + test_size)]
X_no_plane_val = no_plane_images[(train_size + test_size):(train_size + test_size + val_size)]
Y_no_plane_train = np.zeros(X_no_plane_train.shape[0])
Y_no_plane_test = np.zeros(X_no_plane_test.shape[0])
Y_no_plane_val = np.zeros(X_no_plane_val.shape[0])

X_train = np.vstack((X_plane_train, X_no_plane_train))
Y_train = np.concatenate((Y_plane_train, Y_no_plane_train))
X_train, Y_train = shuffle(X_train, Y_train)

X_test = np.vstack((X_plane_test, X_no_plane_test))
Y_test = np.concatenate((Y_plane_test, Y_no_plane_test))
X_test, Y_test = shuffle(X_test, Y_test)

X_val = np.vstack((X_plane_val, X_no_plane_val))
Y_val = np.concatenate((Y_plane_val, Y_no_plane_val))
X_val, Y_val = shuffle(X_val, Y_val)


# In[70]:


def visualize_data(positive_images, negative_images, n=6):
    """Plots some examples of two classes in two rows.
    """
    n_row = 2
    n_col = min(n, positive_images.shape[0])
    for i in range(n_col):
        plt.subplot(n_row, n_col, i + 1)
        a = positive_images[i]
        plt.imshow(a)
    for i in range(n_col):
        plt.subplot(n_row, n_col, (i + 1) + n_col)
        a = negative_images[i]
        plt.imshow(a)


# In[71]:


def cnn(size, n_layers):
    """CNN.
    Args:
        size: size of the input images
        n_layers: number of layers
    """

    # Define hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
        else:
            model.add(Conv2D(nuerons[i], KERNEL))

    model.add(Activation('relu'))

    # Add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model


# In[72]:


image_size = np.asarray([X_train.shape[1], X_train.shape[2], X_train.shape[3]])

# Hyperparamater
N_LAYERS = 4


# In[73]:


# Instantiate the model
model = cnn(size=image_size, n_layers=N_LAYERS)


# In[74]:


# Training hyperparamters
EPOCHS = 80
BATCH_SIZE = 200


# In[75]:


# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(
    monitor='loss', 
    min_delta=0, 
    patience=PATIENCE, 
    verbose=0, 
    mode='auto')


# In[76]:


# TensorBoard callback
LOG_DIRECTORY_ROOT = '.'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)


# In[77]:


# Place the callbacks in a list
callbacks = [early_stopping, tensorboard]


# In[78]:


# Train the model
history = model.fit(
    X_train, Y_train, validation_data=(X_val, Y_val), 
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, 
    verbose=2)


# In[79]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left');


# In[80]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left');


# #### از نمودارهای فوق مشخص از که تا حدود دوره ی سی ام، مدل در حال بهبود است و راجع به داده های آموزشی و داده های اعتبارسنجی خوب عمل می کند، اما پس از آن دقت مدل برای داده های اعتبار سنجی به آرامی کاهش می یابد. یعنی مدل میل به بیش برازش پیدا می کند.

# In[81]:


# Make a prediction on the test set
test_predictions = model.predict(X_test)
test_predictions = np.round(test_predictions)


# In[82]:


# Report the accuracy
accuracy = accuracy_score(Y_test, test_predictions)
print("Unseen test data accuracy: " + str(accuracy))


# In[83]:


# check development (validation) set
val_predictions = model.predict(X_val)
val_predictions = np.round(val_predictions)


# In[84]:


# Report the accuracy
accuracy = accuracy_score(Y_val, val_predictions)
print("Validation set Accuracy : " + str(accuracy))


# #### شبکه ی کانولوشن با دقت قابل قبولی قدرت تشخیص هواپیما را دارد. دقت تشخیص این شبکه برای داده های تستی مشاهده نشده حدود ۹۸ درصد است
