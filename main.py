# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import os
import json
from convert_veracity_annotations import *
import csv
import collections
import argparse
from utils import *
from model import *
from ordered_set import OrderedSet as oset
import requests
from urllib.parse import urlparse
from sklearn.metrics import jaccard_score


def class_value(classV):

    if classV > 5:
        return 1
    elif classV < -5:
        return -1
    else:
        return 0

#url ex. is str '
def getBias(uurl, name, df):
    bias, eVal = 0, 0
    print(uurl, name)
    if df['site_name'].str.contains(name).any():
        Series = df['site_name'].str.contains(name, regex=False)
        eVal = 1
        bias = df.at[Series[Series == True].index[0], 'bias_rating']
        print("name: ", name, ", bias: ", bias,", eVal: ", eVal)

    else:
        try:
            expanded_url = requests.head(uurl, allow_redirects=True, timeout=5).url
        except Exception as e:
            expanded_url = uurl
            print("url exception caught - ", expanded_url, " could not be grabbed")
            return 0, 0
        if expanded_url != "None":
            o = urlparse(expanded_url)
            #print("netloc is: ", o.netloc)
            if df['url'].str.contains(o.netloc, regex=False).any():
                Series = df['url'].str.contains(o.netloc, regex=False)
                #print(Series[Series == True].index[0])
                bias = df.at[Series[Series == True].index[0], 'bias_rating']
                eVal = 1
            elif df['url'].str.contains(expanded_url, regex=False).any():
                eVal = 1
                bias = df.loc[expanded_url]['bias_rating']
            else:
                bias = 0
                eVal = 0
        else:
            bias = 0
            eVal = 0

        print("name: ", name, ", url: ", expanded_url, ", bias: ", bias,", eVal: ", eVal)

    #print(expanded_url)

    return bias, eVal





def readTopic(topicname):
    origpath = "C:\\Users\\minim\\OneDrive\\Desktop\\pheme rumors\\pheme-rumour-scheme-dataset\\threads\\en\\"
    path = "{opath}{db}".format(opath=origpath, db=topicname)
    cols = ['contributors', 'truncated', 'text', 'in_reply_to_status_id', 'id', 'favorite_count',
            'source', 'retweeted', 'coordinates', 'entities', 'in_reply_to_screen_name', 'id_str',
            'retweet_count', 'in_reply_to_user_id', 'favorited', 'user', 'geo',
            'in_reply_to_user_id_str', 'possibly_sensitive', 'lang', 'created_at', 'filter_level',
            'in_reply_to_status_id_str', 'place', 'metadata', 'extended_entities']
    df = pd.DataFrame(columns=cols)

    '''
    what if we made it a dataframe based on each tweet
    
    tweet-id    author-name  author-url  text  retweets list (users)   
    
    and then read the list of who-follows whom into a csv?

    '''
    tweetcols = ['id', 'author-id', 'author-name', 'author-url', 'text', 'retweets', 'label']
    #tweets = pd.DataFrame(columns=tweetcols)
    '''
    this os walk reads as follows:
    
    annotation.json
    images.dat
    retweets.json
    structure.json
    urls.dat
    who-follows-whom.dat
    source-tweets json
    
    repeat with next thread
    
    '''
    label = 0
    verified_data = list()
    unverified_data = list()
    i = 0
    j = 0
    udict = collections.defaultdict(list)

    ts = oset()
    us = oset()

    for subdir, dirs, files in os.walk(path):
        if "images" in subdir or "reactions" in subdir or "urls-content" in subdir:
            pass
        else:
            for filename in files:
                fpath = os.path.join(subdir, filename)
                if filename == "annotation.json":
                    id = int(subdir.split("\\")[-1])
                    f = open(fpath, 'r')
                    dict = json.load(f)
                    label = convert_annotations(dict, string=False)
                    print(label)
                    #print(label) #false =1, true = -1
                    #read annotation to see if it should be put in training or testing (DL/DU)
                    #twrow = pd.DataFrame({'id': id, 'author': '', 'author-url': '', 'text': '', 'retweets': [], 'label': label})
                    if label == 0:
                        unverified_data.append(
                            {'id': id, 'author-id': 0, 'author-name': '', 'author-url': '', 'text': '', 'retweets': [],
                             'label': label})
                    else:
                        verified_data.append(
                            {'id': id, 'author-id': 0, 'author-name': '', 'author-url': '', 'text': '', 'retweets': [],
                             'label': label})
                    f.close()
                if label == 0: #not labelled, must be read seperately into DU folders

                    if filename == "retweets.json":
                        retweets = []
                        #print(id)
                        rtf = open(fpath, 'r')
                        for line in rtf:
                            rtjson = json.loads(line)
                            #towrite = {'id': dict['user']['id'], 'tweet-id-rtd': dict['retweeted_status']['id']}
                            retweets.append(rtjson['user']['id']) #list of users retweeting tweet id=
                        unverified_data[i]['retweets'] = retweets
                        rtf.close()
                    if filename == "who-follows-whom.dat":
                        '''
                        with open(fpath, 'r') as users:
                            
                            userreader = csv.reader(users, delimiter='\t')
                            for row in userreader:
                                udict[int(row[0])].append(int(row[1]))
                                us.add((int(row[0]), int(row[1])))
                                ts.add((int(row[0]), int(row[1])))
                        '''
                        udf = pd.read_csv(fpath, sep='\t| |[|]', engine='python', header=None)
                        for index, value in udf.iterrows():
                            udict[int(value[0])].append(int(value[1]))
                            us.add((int(value[0]), int(value[1])))
                            ts.add((int(value[0]), int(value[1])))
                    if "source-tweets" in subdir:
                        with open(fpath, 'r') as source:
                            dict = json.load(source)
                            dict['text'] = re.sub(r"[\n\t]*", "", dict['text'])
                            dict['text'] = dict['text'].replace('"', "")
                            res = re.sub(r'[^\w\s]', '', dict['text'])


                            #add appropriate column values to dataframe
                            unverified_data[i]['text'] = dict['text']
                            unverified_data[i]['author-id'] = dict['user']['id']
                            unverified_data[i]['author-name'] = dict['user']['name']
                            unverified_data[i]['author-url'] = dict['user']['url']

                        i += 1

                    #print(subdir, filename)
                else:

                #tweet is real or fake -- yay!
                    if filename == "retweets.json":
                        retweets = []
                        rtf = open(fpath, 'r')
                        for line in rtf:
                            rtjson = json.loads(line)
                            retweets.append(rtjson['user']['id'])  # list of users retweeting tweet id=
                        verified_data[j]['retweets'] = retweets
                        rtf.close()
                    if filename == "who-follows-whom.dat":
                        '''
                        with open(fpath, 'r') as users:
                            userreader = csv.reader(users, delimiter='\t')
                            for row in userreader:
                                udict[int(row[0])].append(int(row[1]))
                        '''
                        udf = pd.read_csv(fpath, sep='\t| |[|]', engine='python', header=None)
                        for index, value in udf.iterrows():
                            #print("row item: ", value[0], value[1])
                            udict[int(value[0])].append(int(value[1]))
                            us.add((int(value[0]), int(value[1])))
                            ts.add((int(value[0]), int(value[1])))
                    if "source-tweets" in subdir:
                        with open(fpath, 'r') as source:
                            dict = json.load(source)
                            dict['text'] = re.sub(r"[\n\t]*", "", dict['text'])
                            dict['text'] = dict['text'].strip('"')
                            res = re.sub(r'[^\w\s]', '', dict['text'])

                            # add appropriate column values to dataframe
                            verified_data[j]['text'] = dict['text']
                            verified_data[j]['author-id'] = dict['user']['id']
                            verified_data[j]['author-name'] = dict['user']['name']
                            verified_data[j]['author-url'] = dict['user']['url']

                        j += 1
    training_tweets = pd.DataFrame(verified_data, columns=tweetcols)
    testing_tweets = pd.DataFrame(unverified_data, columns=tweetcols)
    #print(udict)
    userdf = pd.DataFrame(udict.items(), columns=['user-id', 'following'])





#do all of this for the verified tweets

    corpus = training_tweets['text']

    #cps = corpus.to_string(header=False, index=False)
    #print(cps)
    with open('data/{}_corpus.txt'.format(topicname), 'w', encoding='utf-8') as fp:
        for index, value in corpus.items():
            fp.write(value)
            fp.write('\n')


    #print(corpus)
    #make text file for (X) V

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', default='data/{}_corpus.txt'.format(topicname), help='input text file')
    parser.add_argument('--corpus_file', default='data/{}_doc_term_mat.txt'.format(topicname), help='term document matrix file')
    parser.add_argument('--vocab_file', default='data/{}_vocab.txt'.format(topicname), help='vocab file')
    parser.add_argument('--vocab_max_size', type=int, default=10000, help='maximum vocabulary size')
    parser.add_argument('--vocab_min_count', type=int, default=3, help='minimum frequency of the words')
    args = parser.parse_args()

    print('create vocab')
    vocab = {}
    # print("CWD: ", os.getcwd())
    fp = open(args.text_file, 'r', encoding="utf-8")
    for line in fp:
        arr = re.split('\s', line[:-1])
        for wd in arr:
            try:
                vocab[wd] += 1
            except:
                vocab[wd] = 1
    fp.close()
    vocab_arr = [[wd, vocab[wd]] for wd in vocab if vocab[wd] > args.vocab_min_count]
    vocab_arr = sorted(vocab_arr, key=lambda k: k[1])[::-1]
    vocab_arr = vocab_arr[:args.vocab_max_size]
    vocab_arr = sorted(vocab_arr)

    fout = open(args.vocab_file, 'w', encoding="utf-8")
    for itm in vocab_arr:
        itm[1] = str(itm[1])
        fout.write(' '.join(itm) + '\n')
    fout.close()

    # vocabulary to id
    vocab2id = {itm[1][0]: itm[0] for itm in enumerate(vocab_arr)}
    print('create document term matrix')
    data_arr = []
    fp = open(args.text_file, 'r', encoding="utf-8")
    fout = open(args.corpus_file, 'w', encoding="utf-8")
    for line in fp:
        arr = re.split('\s', line[:-1])
        arr = [str(vocab2id[wd]) for wd in arr if wd in vocab2id]
        sen = ' '.join(arr)
        fout.write(sen + '\n')
    fp.close()
    fout.close()



    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', default='data/{}_doc_term_mat.txt'.format(topicname), help='term document matrix file')
    parser.add_argument('--vocab_file', default='data/{}_vocab.txt'.format(topicname), help='vocab file')
    parser.add_argument('--model', default='seanmf', help='nmf | seanmf')
    parser.add_argument('--max_iter', type=int, default=500, help='max number of iterations')
    parser.add_argument('--n_topics', type=int, default=100, help='number of topics')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--beta', type=float, default=0.0, help='beta')
    parser.add_argument('--max_err', type=float, default=0.1, help='stop criterion')
    parser.add_argument('--fix_seed', type=bool, default=True, help='set random seed 0')
    args2 = parser.parse_args()

    docs = read_docs(args2.corpus_file)
    vocab = read_vocab(args2.vocab_file)
    n_docs = len(docs)
    n_terms = len(vocab)
    print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

    tmp_folder = '{}_results'.format(topicname)
    if not os.access(tmp_folder, os.F_OK):
        os.mkdir(tmp_folder)

    if args2.model.lower() == 'nmf':
        print('read term doc matrix')
        dt_mat = np.zeros([n_terms, n_docs])
        for k in range(n_docs):
            for j in docs[k]:
                dt_mat[j, k] += 1.0
        print('term doc matrix done')
        print('-' * 50)
        np.save('{}\\X.npy'.format(tmp_folder), dt_mat)
        '''
        model = NMF(
            dt_mat,
            n_topic=args2.n_topics,
            max_iter=args2.max_iter,
            max_err=args2.max_err)

        model.save_format(
            Wfile=tmp_folder + '/W.txt',
            Hfile=tmp_folder + '/H.txt')
         '''

    if args2.model.lower() == 'seanmf':
        print('calculate co-occurance matrix')
        dt_mat = np.zeros([n_terms, n_terms])
        for itm in docs:
            for kk in itm:
                for jj in itm:
                    dt_mat[int(kk), int(jj)] += 1.0

        print('co-occur done')
        print('-' * 50)
        print('calculate PPMI')
        D1 = np.sum(dt_mat)
        SS = D1 * dt_mat
        for k in range(n_terms):
            SS[k] /= np.sum(dt_mat[k])
        for k in range(n_terms):
            SS[:, k] /= np.sum(dt_mat[:, k])
        dt_mat = []  # release memory
        SS[SS == 0] = 1.0
        SS = np.log(SS)
        SS[SS < 0.0] = 0.0
        print('PPMI done')
        print('-' * 50)

        print('read term doc matrix')
        dt_mat = np.zeros([n_terms, n_docs])
        for k in range(n_docs):
            for j in docs[k]:
                dt_mat[j, k] += 1.0
        print('term doc matrix done')
        print('-' * 50)
        np.save('{}\\X.npy'.format(tmp_folder), dt_mat)
        '''
        model = SeaNMFL1(
            dt_mat, SS,
            alpha=args2.alpha,
            beta=args2.beta,
            n_topic=args2.n_topics,
            max_iter=args2.max_iter,
            max_err=args2.max_err,
            fix_seed=args2.fix_seed)

        model.save_format(
            W1file=tmp_folder + '/W.txt',
            W2file=tmp_folder + '/Wc.txt',
            Hfile=tmp_folder + '/H.txt')
        '''
    # make retweet matrix (W)
    #make publisher-news matrix (B)

    users = []
    for key, value in udict.items():
        users.append(key)
        for u in value:
            users.append(u)
    users = np.unique(users)


    W = np.zeros((len(users), len(training_tweets)))
    P = training_tweets['author-id']
    P = list(P.unique())
    n = len(training_tweets)
    l = len(P)
    B = np.zeros((l, n))
    biases = {}
    eVals = {}
    biasData = pd.read_csv("data/media-bias-scrubbed-results.csv", header=0)
    # i = df['url']
    biasData.set_index(biasData['url'])
    biases = {}
    evals = {}
    print("building B, W, o, e, y")
    y = np.zeros((len(training_tweets), 1))
    for j, value in training_tweets.iterrows():
        k = P.index(value['author-id'])
        B[k, j] = 1

        for i in range(len(users)):
            if users[i] in value['retweets']:
                W[i, j] = 1
                #print("user i retweeted news article j!")
        if value['author-url'] != 'None':
            bias, eVal = getBias(value['author-url'], value['author-name'], biasData)
            bias = class_value(bias)

        else:
            bias = 0
            eVal = 0
        biases[value['author-id']] = (int(bias))
        evals[value['author-id']] = (int(eVal))
        y[j] = value['label']
    #print(y)

    o = np.asarray(list(biases.values()), dtype=np.int32)
    e = np.asarray(list(evals.values()), dtype=np.int32)
    np.save('{}\\o.npy'.format(tmp_folder), o)
    np.save('{}\\y.npy'.format(tmp_folder), y)
    np.save('{}\\e.npy'.format(tmp_folder), e)
    np.save('{}\\W.npy'.format(tmp_folder), W)
    np.save('{}\\B.npy'.format(tmp_folder), B)
    print("B and W done")
    print('-' * 50)


    #make user-user matrix (A)
    A = np.zeros((len(users), len(users)), dtype=np.int8)  # 19906
    print("building A")
    while len(us) > 0:
        it = us.pop()
        #print(it[0], it[1])
        if it[0] in users and it[1] in users:
        #if(it[0] in U):
            if (it[1], it[0]) in ts:
                #print((it[1], it[0]), (it[1], it[0]) in ts)
                #print(np.where(users == it[0])[0][0])
                i = np.where(users == it[0])[0][0]

            #if(it[1] in U):
                j = np.where(users == it[1])[0][0]
                A[i, j] = 1

             #   j = np.where(U == it[0])[0][0]
        #elif(it[1] in U):
           # i = j = np.where(U == it[1])[0][0]
        else:
            pass

        #print(i, j)
        #A[i, j] = 1

    for i in range(len(users)):
        A[i, i] = 0
    #print(A.sum())
    np.save('{}\\A.npy'.format(tmp_folder), A)
    Uset = np.array(users)
    np.save('{}\\U.npy'.format(tmp_folder), Uset)
    print("A done")
    print('-' * 50)
    print("See CredRank.py to build c...")
    print('#' * 50)







    #make o

    #make c (maybe take the rating of tweets that the user retweeted)
    #also makes e



    #already have y, strip it to npy

#repeat for unverified



                    #print(subdir, filename)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    readTopic("sydneysiege")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
