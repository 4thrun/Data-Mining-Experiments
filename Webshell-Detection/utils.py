# -*- coding: utf-8 -*-
import os 
import subprocess
import re 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# load_php_opcode: gets PHP opcode 
# php.exe has to be added to PATH
# find the plugin that matches the specific PHP version:
# http://pecl.php.net/package/vld
def load_php_opcode(php_filename):
    try:
        output = subprocess.check_output(
            ['php.exe', '-dvld.active=1', '-dvld.execute=0', php_filename],
            stderr=subprocess.STDOUT,
        )
        output = str(output, encoding="UTF-8")
        tokens = re.findall(r'\s(\b[A-Z_]+\b)\s', output)
        t = " ".join(tokens)
        return t 
    except:
        return " "
    

# load_php_opcode_from_dir_with_file: saves all opcodes to one file 
def load_php_opcode_from_dir_with_file(dir, file_name):
    print("[+] load PHP opcode from dir: {}".format(dir))
    for root, dirs, files in os.walk(dir):
        for filename in files:
            if filename.endswith('.php'):
                try:
                    full_path = os.path.join(root, filename)
                    file_content = load_php_opcode(full_path)
                    if file_content.strip() != "":
                        with open(file_name, 'a+', encoding='UTF-8') as f:
                            f.write(file_content + "\n")
                except:
                    continue
    print("[*] done")
    f.close()


# get_feature_by_tfidf: returns feature matrix
def get_feature_by_tfidf(input_X, max_features=None):
    cv = CountVectorizer(
        # N-gram
        ngram_range=(3, 3),
        decode_error='ignore',
        max_features=max_features,
        token_pattern=r'\b\w+\b',
        min_df=1,
        max_df=1.0,
    )
    x = cv.fit_transform(input_X).toarray()
    trans = TfidfTransformer(smooth_idf=False)
    trans = trans.fit_transform(x)
    x = trans.toarray()
    return x