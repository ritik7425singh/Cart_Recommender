#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


plt.style.use("ggplot")


# In[3]:


import sklearn
from sklearn.decomposition import TruncatedSVD


# In[5]:


amazon_ratings = pd.read_csv('ratings_Beauty.csv')
amazon_ratings = amazon_ratings.dropna()
amazon_ratings.head()


# In[6]:


amazon_ratings.shape


# In[7]:


popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(10)


# In[8]:


most_popular.head(30).plot(kind = "bar")


# In[12]:


amazon_ratings1 = amazon_ratings.head(100000)


# In[13]:


ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
ratings_utility_matrix.head()


# In[14]:


ratings_utility_matrix.shape


# In[15]:


X = ratings_utility_matrix.T
X.head()


# In[16]:


X.shape


# In[17]:


X1 = X


# In[18]:


SVD = TruncatedSVD(n_components=100)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape


# In[19]:


correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape


# In[27]:


X.index[999]


# In[33]:


i = "B0000530LO"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID


# In[34]:


correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape


# In[40]:


Recommend = list(X.index[correlation_product_ID > 0.80])
Recommend.remove(i) 
Recommend[0:9]


# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# In[42]:


product_descriptions = pd.read_csv('product_descriptions.csv')
product_descriptions.shape


# In[43]:


product_descriptions = product_descriptions.dropna()
product_descriptions.shape
product_descriptions.head()


# In[45]:


product_descriptions1 = product_descriptions.head(50000)
product_descriptions1["product_description"].head(10)


# In[46]:


vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])
X1


# In[49]:


X=X1

kmeans = KMeans(n_clusters = 20, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")
plt.show()


# In[50]:


def print_cluster(i):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


# In[53]:


true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(true_k):
    print_cluster(i)


# In[54]:


def show_recommendations(product):
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    print_cluster(prediction[0])


# In[55]:


show_recommendations("cutting tool")


# In[56]:


show_recommendations("steel drill")


# In[ ]:




