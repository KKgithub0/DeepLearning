import paddle
import paddle.fluid as fluid
from paddle.nn import Linear
from paddle.regularizer import L2Decay
import paddle.nn.functional as F
import numpy as np
import os
import sys
import io
import random

"""
for training ecr ecom ctr model
"""

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='surrogateescape')

# sigmoid
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def cross_entropy(a, y):
    return -y*np.log(a)-(1-y)*np.log(1-a)

def load_data():
    data = []
    fea_num = 0
    set_size = 0
    data_set = set()
    for line in sys.stdin:
        res = line.strip().split('\t')
        fea_num = len(res)

        key = '\t'.join(res[:-1])
        if key in data_set:
            continue
        data_set.add(key)

        set_size += 1
        data.extend(res)


    data = np.array(data).astype('float32')
    #print(set_size, fea_num)
    data = data.reshape([set_size, fea_num])
    #data = data.reshape([fea_num, set_size])
    '''
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0) /data.shape[0]
    for i in range(fea_num - 1):
        #if i > 25:
        #    break
        if maximums[i] == minimums[i]:
            continue
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    '''
    return data

def LoadWordEmbedding():
    words = []
    embeddings = []
    with open('./model.bin', 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(' ')
            if len(arr) != 129:
                continue
            words.append(arr[0])
            embeddings.append([np.float32(x) for x in arr[1:]])
    embeddings = np.asarray(embeddings)
    return words, embeddings

def word_to_index(word1, word2):
    word1 = word1.split('|')
    word2 = word2.split('|')
    word_seg1 = []
    word_seg2 = []
    for seg in word1:
        if seg in words:
            word_seg1.append(np.array(words.index(seg)).astype('int64'))
        else:
            # words_size - 2 means UNK
            word_seg1.append(np.array(len(words) - 2).astype('int64'))
    for seg in word2:
        if seg in words:
            word_seg2.append(np.array(words.index(seg)).astype('int64'))
        else:
            # words_size - 2 means UNK
            word_seg2.append(np.array(len(words) - 2).astype('int64'))
    return np.array(word_seg1).astype('int64'), np.array(word_seg2).astype('int64')

def load_data_new():
    # load data
    data_file = './training_set.part'
    data = []
    labels = []
    BATCH_SIZE = 1024
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) != 3:
                continue

            word1, word2, label = arr

            seg_id1, seg_id2 = word_to_index(word1, word2)

            label = np.array(label).astype('float32')

            data.append([seg_id1, seg_id2])
            labels.append(label)


    data_length = len(data)
    print("data size:{}".format(data_length))

    index_list = list(range(data_length))

    #
    def data_generator():
        random.shuffle(index_list)
        data_list = []
        label_list = []
        for i in index_list:
            data_list.append(data[i])
            label_list.append(labels[i])
            if len(data_list) == BATCH_SIZE:
                yield np.array(data_list, dtype='object'), np.array(label_list, dtype='float32')
                data_list = []
                label_list = []

        # left data
        if len(data_list) > 0:
            yield np.array(data_list, dtype='object'), np.array(label_list, dtype='float32')

    return data_generator

class CtrDNNModel(fluid.dygraph.Layer):
    def __init__(self):
        super(CtrDNNModel, self).__init__()
        #self.fc1 = Linear(in_features = 282, out_features = 512)
        w_param_attrs = fluid.ParamAttr(
                name = "emb_weight",
                learning_rate = 0.01,
                initializer = fluid.initializer.NumpyArrayInitializer(embeddings),
                trainable = True)
        self.embedding = paddle.nn.Embedding(
                num_embeddings = len(words),
                embedding_dim = len(embeddings),
                weight_attr = w_param_attrs,
                sparse = True)
        self.fc1 = Linear(in_features = 256, out_features = 512)
        self.fc2 = Linear(in_features = 512, out_features = 256)
        self.fc3 = Linear(in_features = 256, out_features = 128)
        self.fc4 = Linear(in_features = 128, out_features = 64)
        self.fc5 = Linear(in_features = 64, out_features = 1)
        return

    def forward(self, inputs, label = None):
        # embedding, pooling
        word1 = inputs[:, 0]
        word_vec1 = []
        for word in word1:
            word = paddle.to_tensor(word)
            emb = self.embedding(word)
            emb = sum(emb) / len(emb)
            word_vec1.append(emb.numpy())
        word2 = inputs[:, 1]
        word_vec2 = []
        for word in word2:
            word = paddle.to_tensor(word)
            emb = self.embedding(word)
            emb = sum(emb) / len(emb)
            word_vec2.append(emb.numpy())

        # concat
        word_vec1 = paddle.to_tensor(word_vec1)
        word_vec2 = paddle.to_tensor(word_vec2)
        concat_embed = paddle.concat(x = [word_vec1, word_vec2], axis=-1)

        # dnn
        output1 = F.relu(self.fc1(concat_embed))
        output2 = F.relu(self.fc2(output1))
        output3 = F.relu(self.fc3(output2))
        output4 = F.relu(self.fc4(output3))
        output5 = F.sigmoid(self.fc5(output4))
        output_final = output5
        return output_final[:,0]

class CtrLRModel(fluid.dygraph.Layer):
    def __init__(self):
        super(CtrLRModel, self).__init__()
        self.fc = Linear(in_features = 256, out_features = 1)
        w_param_attrs = fluid.ParamAttr(
                name = "emb_weight",
                learning_rate = 0.01,
                initializer = fluid.initializer.NumpyArrayInitializer(embeddings),
                trainable = True)
        self.embedding = paddle.nn.Embedding(
                num_embeddings = len(words),
                embedding_dim = len(embeddings),
                weight_attr = w_param_attrs,
                sparse = True)
        return

    def forward(self, inputs, label = None):
        word1 = inputs[:, 0]
        word_vec1 = []
        for word in word1:
            word = paddle.to_tensor(word)
            emb = self.embedding(word)
            emb = sum(emb) / len(emb)
            word_vec1.append(emb.numpy())
        word2 = inputs[:, 1]
        word_vec2 = []
        for word in word2:
            word = paddle.to_tensor(word)
            # embedding
            emb = self.embedding(word)
            # pooling
            emb = sum(emb) / len(emb)
            word_vec2.append(emb.numpy())

        # concat
        word_vec1 = paddle.to_tensor(word_vec1)
        word_vec2 = paddle.to_tensor(word_vec2)
        concat_embed = paddle.concat(x = [word_vec1, word_vec2], axis=-1)

        output = self.fc(concat_embed)
        output_final = F.sigmoid(output)
        return output_final[:,0]

#
def model_train2():
    with fluid.dygraph.guard():
        #model = CtrDNNModel()
        model = CtrLRModel()
        model.train()

        training_loader = load_data_new()

        opt = paddle.optimizer.SGD(learning_rate=0.01, weight_decay=L2Decay(0.0001), parameters=model.parameters())
        #opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
        #opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
        #opt = paddle.optimizer.Adagrad(learning_rate=0.0001, parameters=model.parameters())
        #opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

        num_epochs = 100
        for epoch_id in range(num_epochs):
            for batch_id, data in enumerate(training_loader()):
                x, y = data
                y = paddle.to_tensor(y)

                # forward
                predicts = model(x)

                # loss
                loss = F.binary_cross_entropy(predicts, y)
                avg_loss = paddle.mean(loss)

                if batch_id % 100 == 0:
                    print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))


                # backward
                avg_loss.backward()
                opt.step()
                opt.clear_grad()
        paddle.save(model.state_dict(), 'ctr-model.pdparams')

# main
def model_train():
    with fluid.dygraph.guard():
        #model = CtrDNNModel()
        model = CtrLRModel()
        model.train()

        training_data = load_data()

        opt = paddle.optimizer.SGD(learning_rate=0.01, weight_decay=L2Decay(0.0001), parameters=model.parameters())
        #opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
        #opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
        #opt = paddle.optimizer.Adagrad(learning_rate=0.0001, parameters=model.parameters())
        #opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

        num_epochs = 50
        batch_size = 1024
        for epoch_id in range(num_epochs):

            np.random.shuffle(training_data)

            mini_batches = [training_data[k: k + batch_size] for k in range(0, len(training_data), batch_size)]
            for batch_id, mini_batch in enumerate(mini_batches):
                x = np.array(mini_batch[:, :-1]).astype('float32')
                y = np.array(mini_batch[:, -1:]).astype('float32')

                fea_data = fluid.dygraph.to_variable(x)
                click = fluid.dygraph.to_variable(y)

                # forward
                predicts = model(fea_data)

                # loss
                loss = F.binary_cross_entropy(predicts, click)
                avg_loss = paddle.mean(loss)

                if batch_id % 100 == 0:
                    print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))


                # backward
                avg_loss.backward()
                opt.step()
                opt.clear_grad()
        paddle.save(model.state_dict(), 'ctr-model.pdparams')

def model_test():
    with fluid.dygraph.guard():
        model = CtrDNNModel()
        #model = CtrLRModel()
        #model_dict, _ = fluid.load_dygraph('ctr-model.pdparams')
        model_dict = paddle.load('ctr-model.pdparams')
        model.load_dict(model_dict)
        model.eval()

        data = load_data()
        test_data = np.array(data[:, :-1]).astype('float32')
        label = np.array(data[:, -1:]).astype('float32')
        test_data = fluid.dygraph.to_variable(test_data)
        label = fluid.dygraph.to_variable(label)

        result = model(test_data)
        loss = F.binary_cross_entropy(result, label)
        print(paddle.mean(loss))

if __name__ == '__main__':
    global words
    global embeddings
    words, embeddings = LoadWordEmbedding()
    #print('\t'.join(words[:100]))

    if sys.argv[1] == 'train':
        model_train2()
    elif sys.argv[1] == 'test':
        model_test()
