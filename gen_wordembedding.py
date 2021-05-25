# encoding=gbk
import sys
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import jieba

"""
generate word embedding
"""

# 抽取query
def extract_query_from_log(line):
    arr = line.strip().split('\01')
    if len(arr) != 24:
        return []
    ad_pos, query, pre_query, cmatch, words, show_ip, show_time, wosid, wadptid, wbwsid, wosver, wbwsver, wpt, wspeed, sz, wnettype, url, refer, age, gender, cuid, baidu_id, extend_query, query_tradeid = arr
    querys = [query, pre_query, extend_query]
    for word in words.split('\t'):
        if word != '1' and word != '2':
            querys.append(word.split('|')[0])
    return querys

def extract_querys():
    # 抽取query
    query_set = set()
    for line in sys.stdin:
        querys = extract_query_from_log(line)
        for query in querys:
            if query == '':
                continue
            query_set.add(query)
    print('\n'.join(query_set))

# 获取query切词
def get_wordseg(fi):
    query_infos = []
    with open(fi, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            query, seg_infos = line.strip().split('\t')
            segs = []
            for seg_info in seg_infos.split(' '):
                seg, tradeid, level, weight = seg_info.split('@')
                segs.append(seg)
            query_infos.append(segs)
    return query_infos

# 通过jieba切词
def get_wordseg_jieba(fi):
    query_segs = []
    with open(fi, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            query = line.strip()
            if query == '':
                continue
            segs = jieba.cut(query)
            query_segs.append(list(segs))
    return query_segs

# 生成词向量
def gen_word_embedding(query_segs):
    path = get_tmpfile("word2vec.model") #创建临时文件
    sentences = query_segs
    model = Word2Vec(sentences, sg=1, size=128, window=3, min_count=1, negative=3, sample=0.001, hs=1, workers=1)
    model.save("word2vec.model")
    model.wv.save_word2vec_format("model.bin", binary=False)

# 加载模型
def load_model():
    from gensim.models import KeyedVectors
    from gensim.test.utils import datapath
    #model= KeyedVectors.load_word2vec_format(datapath('/home/disk1/xuyikai_rd/JOBS/ysyt/ecr/ecom_ctr/model.bin'), binary=False, unicode_errors='ignore')
    model = Word2Vec.load("word2vec.model")
    #for key in model.similar_by_word(q, topn=10, restrict_vocab=None):
    #    print(key)

def main():
    if sys.argv[1] == 'extract':
        extract_querys()
    elif sys.argv[1] == 'embedding':
        #query_segs = get_wordseg('./word_seg.txt')
        query_segs = get_wordseg_jieba('./words.txt.utf8')
        gen_word_embedding(query_segs)
        #load_model()
    else:
        print('Usage: \n python3 wordembedding.py extract >>> for extract words \n \
                python3 wordembedding.py embedding >>>> for gen wordvec')

if __name__ == '__main__':
    main()
