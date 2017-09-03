#encoding=utf-8
import re
import numpy as np
import pandas as pd
import sys
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

#reload(sys)

import sys  
import importlib
importlib.reload(sys)


word_size = 128
maxlen = 32
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model


#sys.setdefaultencoding("utf-8")  
#print(sys.getdefaultencoding())

#sys.setdefaultencoding('utf-8')

s = open('msr_train.txt','r',encoding='gbk').read()


#s = open('msr_train.txt').read().decode('gbk')
s = s.split('\r\n')

def clean(s): 
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)

data = []
label = []
def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])

for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])
#>>> type(s)
#<class 'list'>
#>>> len(s)
#298229
#>>> s[0]
#'“/s  人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e  '

#>>> type(data)
#<class 'list'>
#>>> len(data)
#298228
#>>> len(label)
#298228

#>>> data[0]
#['“', '人', '们', '常', '说', '生', '活', '是', '一', '部', '教', '科', '书']
#>>> label[0]
#['s', 'b', 'e', 's', 's', 'b', 'e', 's', 's', 's', 'b', 'm', 'e']


d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label

#>>> d['data'][0]
#['“', '人', '们', '常', '说', '生', '活', '是', '一', '部', '教', '科', '书']
#>>> d.shape
#(298228, 2)
#>>> d['data'][0][1]
#'人'
#>>> d['data'][1][1]
#'血'
#>>> d['data'][1]
#['而', '血', '与', '火', '的', '战', '争', '更', '是', '不', '可', '多', '得', '的', '教', '科', '书']


#run d['data'].apply(len)
#298224    50
#298225     5
#298226    12
#298227    27
#Name: data, Length: 298228, dtype: int64

d = d[d['data'].apply(len) <= maxlen]


d.index = range(len(d))

# after run d.index = range(len(d))
#289451               [b, e, s, b, e, s, b, e, b, m, e, s]  
#289452  [s, s, s, b, e, b, e, s, s, b, e, s, s, s, s, ...  
#
#289453 rows x 2 columns]

tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})


chars = []
for i in data:
    chars.extend(i)

#>>> tag.values
#array([1, 3, 2, 0, 4])
#>>> chars = []
#>>> for i in data:
#...     chars.extend(i)
#... 
#>>> type(chars)
#<class 'list'>
#>>> len(chars)
#3752241

chars = pd.Series(chars).value_counts()
#>>> chars = pd.Series(chars).value_counts()
#>>> chars
#的    129755
#一     40390
#国     40301
#在     32755
#中     29953
#了     29460
#鹗         1
#葩         1
#Length: 5162, dtype: int64
#>>> type(chars)
#<class 'pandas.core.series.Series'>


chars[:] = range(1, len(chars)+1)
#>>> chars[:] = range(1, len(chars)+1)
#>>> type(chars)
#<class 'pandas.core.series.Series'>
#>>> chars
#的       1
#一       2
#国       3
#在       4
#中       5
#了       6


from keras.utils import np_utils
#d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
#print(x)
print('d[x]:')
print(d['x'])
#print(d['label'])
#print(d['label'][0])
#print(d['label'].shape)
#m=tag[x].reshape((-1,1))
#n=[np.array([[0,0,0,0,1]])]
#print(type(m))
#print(type(n))
#mm = d['label'].apply(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1))))
#print(mm)
#print(type(mm))
#print(mm.shape)


d['y'] = d['label'].apply(lambda x: np.array(list(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1))))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))
#>>> np_utils.to_categorical(tag[d['label'][0]],5)
#array([[ 1.,  0.,  0.,  0.,  0.],
#       [ 0.,  1.,  0.,  0.,  0.],
#       [ 0.,  0.,  0.,  1.,  0.],
#       [ 1.,  0.,  0.,  0.,  0.],
#       [ 1.,  0.,  0.,  0.,  0.],
#       [ 0.,  1.,  0.,  0.,  0.],
#       [ 0.,  0.,  0.,  1.,  0.],
#       [ 1.,  0.,  0.,  0.,  0.],
#       [ 1.,  0.,  0.,  0.,  0.],
#       [ 1.,  0.,  0.,  0.,  0.],
#       [ 0.,  1.,  0.,  0.,  0.],
#       [ 0.,  0.,  1.,  0.,  0.],
#       [ 0.,  0.,  0.,  1.,  0.]])


print(d['y'])
print(type(d['y']))
print(d['y'].shape)

#d['y'] = d['label'].apply(lambda x: np.array(tag[x].reshape((-1,1))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))


sequence = Input(shape=(maxlen,), dtype='int32')
print(sequence)
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)

#>>> type(d['y'])
#<class 'pandas.core.series.Series'>
#>>> len(d['x'])
#289453
#>>> len(d['label'])
#289453
#>>> sequence = Input(shape=(maxlen,), dtype='int32')
#>>> sequence
#<tf.Tensor 'input_1:0' shape=(?, 32) dtype=int32>
#>>> embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
#>>> embedded
#<tf.Tensor 'embedding_1/Gather:0' shape=(?, 32, 128) dtype=float32>
#>>> blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
#>>> blstm
#<tf.Tensor 'bidirectional_1/add_16:0' shape=(?, ?, 64) dtype=float32>
#>>> 


output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#>>> model.layers
#[<keras.engine.topology.InputLayer object at 0x7f6ba09d2898>, <keras.layers.embeddings.Embedding object at 0x7f6ba09ca470>, <keras.layers.wrappers.Bidirectional object at 0x7f6ba09d2f28>, <keras.layers.wrappers.TimeDistributed object at 0x7f6b9d0cccf8>]


#>>> mm.shape
#(289453, 32, 5)
#>>> type(mm）
#  File "<stdin>", line 1
#    type(mm）
#           ^
#SyntaxError: invalid character in identifier
#>>> type(mm)
#<class 'numpy.ndarray'>
#>>> np.array(list(d['x']))
#array([[  19,    8,   45, ...,    0,    0,    0],
#       [ 109,  851,   83, ...,    0,    0,    0],
#       [ 399,  414,   56, ...,    0,    0,    0],
#       ..., 
#       [4497, 1092,  821, ...,    0,    0,    0],
#       [  27,   72,    4, ...,    0,    0,    0],
#       [ 487,  152,    4, ...,    0,    0,    0]])


batch_size = 1024
history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, nb_epoch=50)

zy = {'be':0.5, 
      'bm':0.5, 
      'eb':0.5, 
      'es':0.5, 
      'me':0.5, 
      'mm':0.5,
      'sb':0.5, 
      'ss':0.5
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}

def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]

def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []

not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result
