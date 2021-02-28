#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:51:44 2021

@author: Waquar Shamsi
"""

import pandas as pd
import os

import pickle
import nltk
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer  
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from os import path


if not path.exists("dataset_ir_ass1"):
    directory = '.'
    os.chdir(directory)
    # GET AND STORE NAME OF ALL FILES, --- NOTE: non .txt files also read
    filelist = os.listdir()
    
    # REMOVE UN-REQUIRED FILE NAMES
    filelist.remove("index.html")
    filelist.remove("SRE")
    filelist.remove("FARNON")
    # print(len(filelist))
    
    dataset_list = []
    i=1 #DOC_ID
    lemmatizer = WordNetLemmatizer() 
    for f in filelist:
      row = []
      with open(directory+f, 'r',encoding='utf-8',errors='ignore') as myfile:
        data=myfile.read().replace('\n', '')
        #TOKENIZE
        data_tokenized = word_tokenize(data)
        #REMOVE STOP WORDS
        stop_words = set(stopwords.words('english'))   
        data_no_sw = [w for w in data_tokenized if not w in stop_words]
        #FILTER OUT PUNCTUATIONS
        data_no_punk = [word for word in data_no_sw if word.isalpha()]
        #LEMMATIZATION
        data_lemma = [lemmatizer.lemmatize(word) for word in data_no_punk]
        #LOWER CASE
        data_low = [word.lower() for word in data_lemma]
        row.append(i)
        i+=1
        row.append(f)
        row.append(data_low)
        dataset_list.append(row)
    
    # READ SRE FOLDER
    directory = 'SRE/'
    os.chdir(directory)
    # GET AND STORE NAME OF ALL FILES, --- NOTE: non .txt files also read
    filelist = os.listdir()
    filelist.remove('index.html')
    filelist.remove('.descs')
    filelist.remove('.header')
    filelist.remove('.musings')
    
    
    i=1 #DOC_ID
    lemmatizer = WordNetLemmatizer() 
    for f in filelist:
      row = []
      with open(directory+f, 'r',encoding='utf-8',errors='ignore') as myfile:
        data=myfile.read().replace('\n', '')
        #TOKENIZE
        data_tokenized = word_tokenize(data)
        #REMOVE STOP WORDS
        stop_words = set(stopwords.words('english'))   
        data_no_sw = [w for w in data_tokenized if not w in stop_words]
        #FILTER OUT PUNCTUATIONS
        data_no_punk = [word for word in data_no_sw if word.isalpha()]
        #LEMMATIZATION
        data_lemma = [lemmatizer.lemmatize(word) for word in data_no_punk]
        #LOWER CASE
        data_low = [word.lower() for word in data_lemma]
        row.append(i)
        i+=1
        row.append(f)
        row.append(data_low)
        dataset_list.append(row)
    
    dataset_df = pd.DataFrame(dataset_list,columns=('doc_id','filename','content'))
    
    print(dataset_df)
    
    #TODO : ADD FILES FROM THAT OTHER FOLDER AS WELL
    #TODO : APPLY PRE PROCESSING - same preprocessing on queries as that on data
    #TODO : CHOOSE BETWEEN LEMMETIZING AND STEMMING
    #TODO : Should I have or not have stop words
    dataset_pickle = open('dataset_ir_ass1','wb')
    pickle.dump(dataset_df, dataset_pickle)
    dataset_pickle.close()
else:
    dataset_pickle = open('dataset_ir_ass1','rb')
    dataset_df = pickle.load(dataset_pickle)
    print(dataset_df)
    
#NOW CREATE THE Dictionary and Posting
term_vs_doc_id = []
for index, row in dataset_df.iterrows():
  content = row['content']
  doc_id = row['doc_id']
  for word in content:
    entry = []
    entry.append(word)
    entry.append(doc_id)
    term_vs_doc_id.append(entry)

term_vs_doc_id_df = pd.DataFrame(term_vs_doc_id,columns=('term','doc_id'))
print(term_vs_doc_id_df)

 #drop duplicate rows
term_vs_doc_id_df_no_dup = term_vs_doc_id_df.drop_duplicates()
print(term_vs_doc_id_df_no_dup)
 
 # sort first by term and then by doc_id
term_vs_doc_id_df_no_dup.sort_values(by=['term','doc_id'])
all_doc_ids = term_vs_doc_id_df_no_dup["doc_id"]
all_doc_ids = all_doc_ids.unique()
# WHAT TO DO ABOUT THESE SINGLE CHARACTER WORDS - arent they stop words

# two dictionaries- one for term vs frequency, another for term vs postings
# DICTIONARY 1 - Term vs Frequency, INITIAL VALUE SET AS 1
term_freq = dict.fromkeys(term_vs_doc_id_df_no_dup['term'].unique(), 0)
term_posting = dict.fromkeys(term_vs_doc_id_df_no_dup['term'].unique(),[]) # DEFAULT VALUE NONE
print(len(term_posting['well']))


term1 =[]
post= [[]]
ddd=dict()
i=0
for index, row in term_vs_doc_id_df_no_dup.iterrows():
  term = row['term']
  # print(ddd)
  doc_id = row['doc_id']
  if term in ddd:
    if doc_id not in ddd[term]:
      ddd[term].append(doc_id)

  else:
    ddd[term]=[doc_id]


  # term_freq[term] += 1
  # print(term_posting[term])
  # if doc_id not in term_posting[term]:
  #   # term_posting[term] = term_posting[term].append(doc_id)
  #   term_posting[term].append(doc_id)
  #   print(term_posting[term])
  # if i<100:
  #   # print(term,doc_id)
  #   print(ddd)
    # print(term_posting[term])
  i+=1
  # if i==100:
  #   break
  
  
d_frew=dict()
for tt in ddd:
  d_frew[tt] = len(ddd[tt])

if not path.exists("posting.pkl"):
    a_file = open("posting.pkl", "wb")
    pickle.dump(ddd, a_file)
    a_file.close()
else: 
    a_file = open("posting.pkl", "rb")
    postings = pickle.load(a_file)
    a_file.close()
    
if not path.exists("frequency.pkl"):
    a_file = open("frequency.pkl", "wb")
    pickle.dump(d_frew, a_file)
    a_file.close()
else:
    a_file = open("frequency.pkl", "rb")
    frequency = pickle.load(a_file)
    a_file.close()
 
def MERGE_AND(term_posting_1, term_posting_2):
  comparisons = 0
  s1 = len(term_posting_1)
  s2 = len(term_posting_2)
  result = []
  ''' IF ELSE TO CHECK WHICH POSTING LIST IS SMALLER TO INITIALIZE RESULT WITH THAT '''
  j=0
  i=0
  while i<s1 and j<s2:
    # shoud i add no. of comparisons with each if or 
    # with each iteration
    comparisons += 1
    if term_posting_2[j] == term_posting_1[i]:
      result.append(term_posting_2[j])
      i+=1
      j+=1
    elif term_posting_2[j] < term_posting_1[i]:
      j+=1
    else:
      i+=1
  return result, comparisons

def MERGE_NOT(term_posting, universal_list):
  # all docs id list - term posting 
  # should i count the number of comparisons made for NOT operation
  # print("POSTING ",term_posting)
  # print("UNIVERSAL LIST",universal_list)
  result = [x for x in universal_list if x not in term_posting]
  # print("RESULT",result)
  return result, 0

def MERGE_OR(term_posting_1, term_posting_2):
  comparisons = 0
  s1 = len(term_posting_1)
  s2 = len(term_posting_2)
  result = []
  ''' IF ELSE TO CHECK WHICH POSTING LIST IS SMALLER TO INITIALIZE RESULT WITH THAT '''
  j=0
  i=0
  while i<s1 or j<s2:
    if i>=s1:
      while(j<s2):
        result.append(term_posting_2[j])
        j+=1
    elif j>=s2:
      while(i<s1):
        result.append(term_posting_1[i])
        i+=1
    elif term_posting_2[j] == term_posting_1[i]:
      result.append(term_posting_1[i])
      i+=1
      j+=1
    elif term_posting_2[j] < term_posting_1[i]:
      result.append(term_posting_2[j])
      j+=1
    else:
      result.append(term_posting_1[i])
      i+=1
    comparisons += 1
  return result, comparisons



def get_stats(c_query, term_posting, term_freq):
  total_comparisons = 0
  query_array = c_query.split(" ")
  i=1
  posting1 = term_posting[query_array[0]]
#  print("Word",query_array[0],"\n Postings: ",posting1)
  while i<len(query_array):
    print(query_array[i])
    e = False
    if query_array[i] in ["AND","OR","NOT"]:
      if query_array[i+1] == "NOT":
        #then take 
        posting2,comparisons = MERGE_NOT(term_posting[query_array[i+2]],all_doc_ids)
#        print("Word2",query_array[i+2],"\n Postings: ",posting2)
        e = True
      else:
        posting2 = term_posting[query_array[i+1]]
#        print("Word2",query_array[i+1],"\n Postings: ",posting2)
    if query_array[i]=="AND":
      posting1, comparisons = MERGE_AND(posting1,posting2)
#      print("\nintermediate",query_array[i+1],"\n Postings: ",posting1)
      total_comparisons += comparisons
    elif query_array[i]=="OR":
      posting1, comparisons = MERGE_OR(posting1,posting2)
#      print("\nintermediate",query_array[i+1],"\n Postings: ",posting1)
      total_comparisons += comparisons
    i+=2 # GOTO NEXT OPERATOR
    if e:
      i+=1
  num_of_docs_matched = len(posting1)
  
  return num_of_docs_matched, total_comparisons


while True:
  query = input("ENTER THE INPUT QUERY:\t")
  operation_seq = input("ENTER THE OPERATION SEQUENCE:\t")
  #HANDLE COMMAS
  query = query.split(",")
  query = [x.strip() for x in query ]
  query = ' '.join(query)
  query = query.lower()

  operation_seq = operation_seq.split(",")
  operation_seq = [x.strip() for x in operation_seq ]
  operation_seq = ' '.join(operation_seq)
  operation_seq =  operation_seq.upper()
  
  stop_words = set(stopwords.words('english'))  
  word_tokens = word_tokenize(query)    
  
  p_query = [w for w in word_tokens if not w in stop_words] 
  p_query = [p.lower() for p in p_query]
  p_query = ' '.join(p_query)
  
  operation_seq  = operation_seq.split(" ")
  #handle NOTs
  i = 0 
  for _ in operation_seq:
    if operation_seq[i] == 'NOT':
      operation_seq[i-1] += " " + operation_seq[i]
      operation_seq.pop(i)
    i+=1

  print(len(p_query.split(" "))-1, len(operation_seq))
  if len(p_query.split(" "))-1 != len(operation_seq):
    print("Number of Operations isn't matching number of words in query!!!")
    continue
  combined_query = ' '.join(' '.join (f for f in tup) for tup in (zip(query.split(" "),operation_seq))) + ' ' + p_query.split()[-1]
  documents_matched, comparisons_made = get_stats(combined_query,postings,frequency)
  print("QUERY : ", combined_query)
  print("Number of Documents Matched: ",documents_matched)
  print("Number of Comparisons Required: ",comparisons_made)
  print("\n")
  ##  lion stood thoughtfully for a moment
  ## OR OR OR
  # telephone paved roads
  # OR NOT AND NOT
