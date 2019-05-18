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
from email_split import email_split
import clearbit
from urlparse import urlparse
import os

clearbit.key = 'sk_c503d1799f222d4838ba71a4f2739e12'

try:
    targetAccountFile = sys.argv[1]
    matchFile = sys.argv[2]
    matchPercentage = float(sys.argv[3])/100
except Exception as e:
    print 'Error with system arguments: ' + str(e)

#targetAccounts = pd.read_csv(goldenFile)
#matchList = pd.read_csv(matchFile)


class Files:
    def __init__(self, targetAccountFile, matchFile):
        self.targetAccountsFrame = pd.read_csv(targetAccountFile)
        self.matchFrame = pd.read_csv(matchFile)


class DataCleaning:
    def cleanMatches(self,  matchFrame):
        stopwords = {'the'}
        for index, row in matchFrame.iterrows():
            tempString = cleanco(matchFrame.iloc[index]['company'].lower()).clean_name()
            resultwords  = [word for word in re.split("\W+",tempString) if word.lower() not in stopwords]
            result = ' '.join(resultwords)
            matchFrame.at[index,'Clean Company'] = result.translate(None, string.punctuation)
            try:
                matchFrame.at[index,'Email Domain'] = email_split(row['email']).domain
            except Exception as e:
                print 'error ' + str(e)
                print 'no email'
                pass

        
    def cleanTargetAccounts(self, targetAccountsFrame):
        stopwords = {'the'}
        for index, row in targetAccountsFrame.iterrows():
            tempString = cleanco(targetAccountsFrame.iloc[index]['Account Name'].lower()).clean_name()
            resultwords  = [word for word in re.split("\W+",tempString) if word.lower() not in stopwords]
            result = ' '.join(resultwords)
            targetAccountsFrame.at[index,'Clean Target'] = result.translate(None, string.punctuation)
            try:
                if 'www.' in row['Website']:
                    targetAccountsFrame.at[index,'Website Host'] = urlparse(row['Website']).path.split('.')[1].lower()
                else:
                    targetAccountsFrame.at[index,'Website Host'] = urlparse(row['Website']).path.split('.')[0].lower()

            except Exception as e:
                targetAccountsFrame.at[index,'Website Host'] = 'None'
                pass
        
class FindMatches:

    def createVectors(self,targetAccountsFrame, matchFrame):
        def ngrams(string, n=2):
            x = cleanco(str(string).lower()).clean_name()
            ngrams = zip(*[x[i:] for i in range(n)])
            return [''.join(ngram) for ngram in ngrams]
        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

        self.tf_idf_matrix_matches = vectorizer.fit_transform(matchFrame['Clean Company'])
        self.tf_idf_matrix_targets = vectorizer.transform(targetAccountsFrame['Clean Target'])

    def awesome_cossim_top(self, A, B, ntop, lower_bound=0):
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

        self.matrix = csr_matrix((data,indices,indptr),shape=(M,N))

    def get_matches_df(self, sparse_matrix, A, B, C, D, top=0):
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
        cleanCompany = np.empty([nr_matches], dtype=object)
        similairity = np.zeros(nr_matches)
        emailDomain = np.empty([nr_matches], dtype=object)
        showStatus = np.empty([nr_matches], dtype=object)
        notes = np.empty([nr_matches], dtype=object)
        attendeeType = np.empty([nr_matches], dtype=object)
        
        print(nr_matches)
        for index in range(0, nr_matches):
            #left_side[index] = A[sparserows[index]]
            left_side[index] = C.loc[sparserows[index], 'company']
            cleanCompany[index] = A[sparserows[index]]
            firstName[index] = C.loc[sparserows[index], 'first_name']
            lastName[index] = C.loc[sparserows[index], 'last_name']

            try:
                attendeeType[index] = C.loc[sparserows[index], 'primarycategorization']
            except Exception as e:
                print 'no attendee type'
                pass
            try:
                phone[index] = C.loc[sparserows[index], 'phone']
            except Exception as e:
                print 'no phone'
                pass

            try:
                showStatus[index] = C.loc[sparserows[index], 'capture_type']
            except Exception as e:
                print 'no status'
                pass

            try:
                notes[index] = C.loc[sparserows[index], 'notes']

            except Exception as e:
                print 'no notes'
                pass

            try:
                email[index] = C.loc[sparserows[index], 'email']
                emailDomain[index] = C.loc[sparserows[index], 'Email Domain']
            except Exception as e:
                print 'no email'
                pass
            
            try:
                title[index] = C.loc[sparserows[index], 'title']
            except Exception as E:
                print 'no title found'
                pass

            if D.loc[sparsecols[index], 'Website Host'].lower() == C.loc[sparserows[index], 'Email Domain'].split('.')[0].lower():
                print 'matched company email to domain'
                right_side[index] = B[sparsecols[index]]
                accountId[index] = D.loc[sparsecols[index], 'Account ID']
                accountOwner[index] = D.loc[sparsecols[index], 'Account Owner']
                ownerId[index] = D.loc[sparsecols[index], 'User ID (Adult)']
                similairity[index] = 1

            elif sparse_matrix.data[index] > matchPercentage:
                print 'matched company via fuzzy match'
                right_side[index] = B[sparsecols[index]]
                accountId[index] = D.loc[sparsecols[index], 'Account ID']
                accountOwner[index] = D.loc[sparsecols[index], 'Account Owner']
                ownerId[index] = D.loc[sparsecols[index], 'User ID (Adult)']
                similairity[index] = sparse_matrix.data[index]

            else:
                right_side[index] = 'none'
                accountId[index] = 'none'
                accountOwner[index] = 'none'
                ownerId[index] = 'none'
                similairity[index] = 0


        self.matches_df = pd.DataFrame({'company': left_side,
                            'clean company' : cleanCompany,
                            'matched_company': right_side,
                            'title': title,
                            'first_name':firstName,
                            'last_name':lastName,
                            'phone':phone,
                            'email':email,
                            'email_domain' : emailDomain,
                            'Account Id':accountId,
                            'Account Owner':accountOwner,
                            'Owner Id':ownerId,
                            'Match Similarity': similairity,
                            'Attendee Type' : attendeeType,
                            'Capture Type' : showStatus,
                            'notes' : notes
                            })

class Enrichment:
    companyDictionary = {}
    def clearbitLookup(self, matches_df):
        print 'beginning enrichment'
        companyDictionary = {}
        totalAPICalls = 0

        for index,row in matches_df.iterrows():
            try: 
                if row['email'] != None and 'gmail.com' not in row['email']:
                    if row['email_domain'] in companyDictionary:
                        print 'company exists! WAHOO!'
                        matches_df.at[index,'clearbit_matched_company_domain'] = companyDictionary[row['email_domain']]['domain']
                        matches_df.at[index,'annual revenue'] = companyDictionary[row['email_domain']]['metrics']['annualRevenue']
                        try:
                            matches_df.at[index,'estimated annual revenue'] = str(companyDictionary[row['email_domain']]['metrics']['estimatedAnnualRevenue'])
                        except Exception as e:
                            print 'no estimated revenue'
                            pass
                        matches_df.at[index,'employees'] = companyDictionary[row['email_domain']]['metrics']['employees']
                        matches_df.at[index,'industry'] = companyDictionary[row['email_domain']]['category']['industry']
                        matches_df.at[index,'parent legal name'] = companyDictionary[row['email_domain']]['legalName']
                        matches_df.at[index,'parent_company_domain'] = companyDictionary[row['email_domain']]['parent']['domain']
                        continue
                    else:
                        print 'company doesnt exists. searching on email domain...' + str(row['email_domain'])
                        tempCompany = clearbit.Company.find(domain=row['email_domain'],stream=True)
                        totalAPICalls += 1
                        if tempCompany != None:
                            print 'company exists - adding'
                            companyDictionary[tempCompany['domain']] = tempCompany
                            print tempCompany
                            print 'successfully added company'
                            matches_df.at[index,'clearbit_matched_company_domain'] = companyDictionary[tempCompany['domain']]['domain']
                            print 'wtf 1'
                            matches_df.at[index,'annual revenue'] = companyDictionary[tempCompany['domain']]['metrics']['annualRevenue']
                            print 'wtf 2'
                            try:
                                matches_df.at[index,'estimated annual revenue'] = str(companyDictionary[tempCompany['domain']]['metrics']['estimatedAnnualRevenue'])
                            except Exception as e:
                                print 'no estimated revenue'
                                pass
                            matches_df.at[index,'employees'] = companyDictionary[tempCompany['domain']]['metrics']['employees']
                            matches_df.at[index,'industry'] = companyDictionary[tempCompany['domain']]['category']['industry']
                            matches_df.at[index,'parent legal name'] = str(companyDictionary[tempCompany['domain']]['legalName'])
                            try:
                                matches_df.at[index,'parent_company_domain'] = str(companyDictionary[tempCompany['domain']]['parent']['domain'])
                            except Exception as e:
                                print 'error with parent company domain'
                            tempCompany = {}
                        elif tempCompany == None:
                            print 'company not found... '
                            continue
                else: 
                    pass
            except Exception as e: 
                print 'error ' + str(e)
                print 'no email address - finding match via company name'
                try:
                    response = clearbit.NameToDomain.find(name=row['clean_company'])
                    
                    if response != None and response['domain'] not in companyDictionary:
                        print 'company not in dictionary. searching'
                        tempCompany = clearbit.Company.find(domain=response['domain'],stream=True)
                        totalAPICalls += 1
                        if tempCompany != None:
                            print 'we found a match'
                            companyDictionary[tempCompany['domain']] = tempCompany
                            print tempCompany
                            matches_df.at[index,'clearbit_matched_company_domain'] = companyDictionary[tempCompany['domain']]['domain']
                            matches_df.at[index,'annual revenue'] = companyDictionary[tempCompany['domain']]['metrics']['annualRevenue']
                            matches_df.at[index,'employees'] = companyDictionary[tempCompany['domain']]['metrics']['employees']
                            matches_df.at[index,'industry'] = companyDictionary[tempCompany['domain']]['category']['industry']
                            matches_df.at[index,'parent legal name'] = companyDictionary[tempCompany['domain']]['legalName']
                            if companyDictionary[tempCompany['domain']]['parent']['domain'] != None:
                                matches_df.at[index,'parent_company_domain'] = companyDictionary[tempCompany['domain']]['parent']['domain']
                            tempCompany = {}
                        elif tempCompany == None:
                            print 'no match found'
                            continue
                    elif response != None and response['domain'] in companyDictionary:
                        print 'company exists in dictionary! WAHOO!'
                        matches_df.at[index,'clearbit_matched_company_domain'] = companyDictionary[response['domain']]['domain']
                        matches_df.at[index,'annual revenue'] = companyDictionary[response['domain']]['metrics']['annualRevenue']
                        try:
                            matches_df.at[index,'estimated annual revenue'] = str(companyDictionary[response['domain']]['metrics']['estimatedAnnualRevenue'])
                        except Exception as e:
                            print 'error pulling revenue from company dictionary'
                            pass
                        matches_df.at[index,'employees'] = companyDictionary[response['domain']]['metrics']['employees']
                        matches_df.at[index,'industry'] = companyDictionary[response['domain']]['category']['industry']
                        matches_df.at[index,'parent legal name'] = companyDictionary[response['domain']]['legalName']
                        matches_df.at[index,'parent_company_domain'] = str(companyDictionary[response['domain']]['parent']['domain'])
                        pass
                    else:
                        print response
                        print 'no response'
                        continue
                except:
                    print 'we gots an error'
                    time.sleep(3)
                    pass
        print str(totalAPICalls)



t1 = time.time()

inputs = Files(targetAccountFile,  matchFile)
dataCleaning = DataCleaning()
findMatches = FindMatches()
enrichment = Enrichment()


dataCleaning.cleanMatches(inputs.matchFrame)

dataCleaning.cleanTargetAccounts(inputs.targetAccountsFrame)

findMatches.createVectors(inputs.targetAccountsFrame, inputs.matchFrame)
findMatches.awesome_cossim_top(findMatches.tf_idf_matrix_matches, findMatches.tf_idf_matrix_targets.transpose(), 1, 0)
findMatches.get_matches_df(findMatches.matrix, inputs.matchFrame['Clean Company'], inputs.targetAccountsFrame['Clean Target'], inputs.matchFrame, inputs.targetAccountsFrame)
enrichment.clearbitLookup(findMatches.matches_df)

t = time.time()-t1
print("SELFTIMED:", t)
string = 'outputFile_' + str(time.time()) + '.csv'
findMatches.matches_df.to_csv(string, encoding = 'utf-8')

    

