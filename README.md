<h1>Company Dedup</h1>
<p>This matches companies in one file to a 'target list' in another csv. The primary use case was to match tradeshow and events lists against a target account list.</p>
<p>We tried multiple approaches, including brute force / fuzzy matching across the whole dataset, but found the following approach gave the best results, fastest. It can match roughly 2,200 companies against a 259 </p>
<p>This approach: , , then .</p>
<ul>
<li>Take three command line arguments: target-account-list, file-to-match, minimum-similarity-score</li>
<li>Pre Processes the Company Names: Remove llc, inc, punctuation</li>
<li>Vectorizes the company names as 3 letter n-grams</li>
<li>Places them in a sparse matrix</li>
<li>Does sparse matrix multiplication</li>
<li>oOutputs the matches to a csv where only those with a high enough similarity are shown</li>
</ul>
<h2>Required Packages</h2>
<ul>
<li>import sys</li>
<li>import csv</li>
<li>import pandas as pd</li>
<li>import time</li>
<li>from cleanco import cleanco</li>
<li>from sklearn.feature_extraction.text import TfidfVectorizer</li>
<li>import numpy as np</li>
<li>from scipy.sparse import csr_matrix</li>
<li>import sparse_dot_topn.sparse_dot_topn as ct</li>
<li>import time</li>
<li>import string</li>
</ul>
<h2>Examples</h2>
<p>python match-companies.py targetAccounts.csv sampleFile.csv 95</p>

