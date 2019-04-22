<h1>Company Dedup</h1>
<p>This matches companies in one file to a 'target list' in another csv. The primary use case was to match tradeshow and events lists against a target account list.</p>
<p>We tried multiple approaches, including brute force / fuzzy matching across the whole dataset, but found the following approach gave the best results, fastest.</p>
<h2>Sample Performance</h2>
<p>This performed roughly 65 times faster than our brute force / levenshtein distance on the same dataset</p>
<p>3.5 seconds to match 2,220 names against 250 companies</p>
<h2>How It Works</h2>
<p>This approach: </p>
<ul>
<li>Takes three command line arguments: target-account-list, file-to-match, minimum-similarity-score</li>
<li>Pre Processes the Company Names:
    <ul>
        <li>Removes all punctuation</li>
        <li>Transforms string to lowercase</li>
        <li>Uses CleanCo module to extract clean company name (Removes llc, inc, incorprated, corp corporation etc...)</li>
        </ul>
</li>
<li>Vectorizes the company names as 3 letter n-grams</li>
<li>Places them in a sparse matrix</li>
<li>Does sparse matrix multiplication</li>
<li>Outputs the matches to a csv where only those with a high enough similarity are shown (see system argument above</li>
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
<h2>Example</h2>
<p>python match-companies.py targetAccounts.csv sampleFile.csv 95</p>

