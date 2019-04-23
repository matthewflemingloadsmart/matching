import sys
import csv
import pandas as pd
import time
from cleanco import cleanco
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import time
import string
import re

try:
    goldenFile = sys.argv[1]
    matchFile = sys.argv[2]
    matchPercentage = int(sys.argv[3])/100

except Exception as e:
    print 'Error with system arguments: ' + str(e)

targetAccounts = pd.read_csv(goldenFile)
matchList = pd.read_csv(matchFile)

def ngrams(string, n=2):
    x = cleanco(str(string).lower()).clean_name()
    ngrams = zip(*[x[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

def get_matches_df(sparse_matrix, A, B, C, D, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    firstName = np.empty([nr_matches], dtype=object)
    lastName = np.empty([nr_matches], dtype=object)
    phone = np.empty([nr_matches], dtype=object)
    email = np.empty([nr_matches], dtype=object)
    accountId = np.empty([nr_matches], dtype=object)
    accountOwner = np.empty([nr_matches], dtype=object)
    ownerId = np.empty([nr_matches], dtype=object)
    title = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = A[sparserows[index]]
        right_side[index] = B[sparsecols[index]]
        firstName[index] = C.loc[sparserows[index], 'first_name']
        lastName[index] = C.loc[sparserows[index], 'last_name']
        phone[index] = C.loc[sparserows[index], 'phone']
        email[index] = C.loc[sparserows[index], 'email']
        title[index] = C.loc[sparserows[index], 'title']
        accountId[index] = D.loc[sparsecols[index], 'Account ID']
        accountOwner[index] = D.loc[sparsecols[index], 'Account Owner']
        ownerId[index] = D.loc[sparsecols[index], 'User ID (Adult)']
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'original_company': left_side,
                         'matched_company': right_side,
                         'title': title,
                         'first_name':firstName,
                         'last_name':lastName,
                         'phone':phone,
                         'email':email,
                         'Account Id':accountId,
                         'Account Owner':accountOwner,
                         'Owner Id':ownerId,
                         'similairity': similairity})
t1 = time.time()

stopwords = {'the'}

for index, row in matchList.iterrows():
    tempString = cleanco(matchList.iloc[index]['company'].lower()).clean_name()
    resultwords  = [word for word in re.split("\W+",tempString) if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    matchList.at[index,'Clean Company'] = result.translate(None, string.punctuation)

for index, row in targetAccounts.iterrows():
    tempString = cleanco(targetAccounts.iloc[index]['Account Name'].lower()).clean_name()
    resultwords  = [word for word in re.split("\W+",tempString) if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    targetAccounts.at[index,'Clean Target'] = result.translate(None, string.punctuation)

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix_matches = vectorizer.fit_transform(matchList['Clean Company'])
tf_idf_matrix_targets = vectorizer.transform(targetAccounts['Clean Target'])

matches = awesome_cossim_top(tf_idf_matrix_matches, tf_idf_matrix_targets.transpose(), 1, 0)
t = time.time()-t1
print("SELFTIMED:", t)

matches_df = get_matches_df(matches, matchList['Clean Company'], targetAccounts['Clean Target'], matchList, targetAccounts, top=0)
matches_df = matches_df.loc[(matches_df['similairity'] > matchPercentage)]

matches_df.to_csv('outputFile.csv')

    

