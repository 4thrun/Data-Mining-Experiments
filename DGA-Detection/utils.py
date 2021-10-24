import os 
from math import log2, log10 
import re 
import os 


# for dga-domain.txt only 
def txt2csv(read_path, write_path):
    if not os.path.exists(read_path):
        return False
    f1 = open(file=read_path, mode='r', encoding='UTF-8') 
    f2 = open(file=write_path, mode='w', encoding='UTF-8')
    count = 0
    while True:
        line = f1.readline()
        if not line:
            break 
        if ("#" in line) or line.strip() == '':
            continue
        else:
            count += 1
            tmp = line.split()
            label = tmp[0]
            domain = tmp[1]
            f2.write("{},{},{}\n".format(count, domain, label))
    f1.close()
    f2.close()
    return True 


# length of domain
def domain_length(string):
    return len(string)


# entropy of domain information 
def domain_entropy(string):
    total = len(string)
    letter_cnt = dict()
    for ch in string:
        letter_cnt[ch] = letter_cnt.get(ch, 0) + 1
    
    ent = 0 
    for cnt in letter_cnt.values():
        p=  cnt / total
        ent -= p * log2(p)
    return ent 


# consonants
consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']
# vowels
vowels = ["a", "u", "i", 'o', 'e']


# proportion of consonants 
def conso_percent(domain):
    domain = domain.lower()
    i = 0
    j = 0
    cnt = 0
    length = len(domain)
    while j < length and i<length:
        if i<length and domain[i] not in consonants:
            i += 1
        else:
            j = i + 1
            while j < length and domain[j] in consonants:
                j += 1
            if j > i+1:
                cnt += 1

            i = j + 1
    # print(cnt)
    # print(length)
    return (cnt / length)


# proportion of vowels
def vowels_persent(domain):
    domain=domain.lower()
    cnt = 0
    length = len(domain)
    for ch in domain:
        if ch in vowels:
            cnt += 1
    return (cnt / length)


# proportion of digits 
def digits_persent(domain):
    cnt = 0
    length = len(domain)
    for ch in domain:
        if ch.isdigit():
            cnt = cnt + 1
    return cnt / length 


# n-grams: n=2 
n = 2 
data_path = "./data/"
corpus = os.path.join(data_path, 'corpus.txt')
class NGrams:
    # word frequence 
    word_freq = dict()
    # phrase frequence 
    phr_freq = dict()
    
    def __init__(self, filename) -> None:
        if not os.path.exists(corpus):
            self.append_s('')
        else:
            fp = open(file=filename, mode="r", encoding='UTF-8')
            while True:
                line = fp.readline()
                if not line:
                    break 
                if line.strip() == '':
                    continue 
                self.append_s(line)
            fp.close()
    
    def append_s(self, content):
        # remove blanks and punctuations
        content = ''.join([c for c in content if c.isalnum()])
        content = content.lower()
        length = len(content)
        if length <= n:
            return
        else:
            word_set = self.get_n_gram_set(content)
        # print(word_set)
        keys = list()
        for word in word_set:
            # update word frequence 
            for k in word:
                self.word_freq[k] = self.word_freq.get(k, 0) + 1

            # update phrase frequence 
            key = '%s%s' % (word[0], word[1])
            keys.append(key)
            self.phr_freq[key] = self.phr_freq.get(key, 0) + 1
    
    def get_n_gram_set(self, txt):
        return set(zip(*[txt[i:] for i in range(n)]))
    
    def get_score(self, txt):
        if len(txt) < n:
            return 1
        word_set = self.get_n_gram_set(txt)
        score = 1
        for w in word_set:
            w1 = w[0]
            key = '%s%s' % (w[0], w[1])
            c_cnt = 0

            # word frequence of `w1`
            w1_freq = self.word_freq.get(w1, 0)
            # phrase frequence of `w`
            w_freq = self.phr_freq.get(key, 0)

            # number of phrases starting with `w1`
            words = self.phr_freq.keys()
            for word in words:
                if word[0] == w1:
                    c_cnt += 1
            # phrase frequence of `w`
            # Laplacian smoothing
            prob = (w_freq+1)/(w1_freq+c_cnt)
            score = score*prob
        return score
    
    def get_domain_score(self, domain):
        domain = domain.lower()
        sub_str_list = re.split(r'\.|-', domain)
        score = 1
        for sub_str in sub_str_list:
            score *= self.get_score(sub_str)
        return score
    

