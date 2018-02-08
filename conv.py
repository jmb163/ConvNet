#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import math
import random
import signal, os, sys

datafiles = ['/Users/Jon/Documents/Workspace/Machine_Learning/research/data/t10k-images-idx3-ubyte',
'/Users/Jon/Documents/Workspace/Machine_Learning/research/data/t10k-labels-idx1-ubyte',
'/Users/Jon/Documents/Workspace/Machine_Learning/research/data/train-images-idx3-ubyte',
'/Users/Jon/Documents/Workspace/Machine_Learning/research/data/train-labels-idx1-ubyte']

trainImages = datafiles[2]
trainLabels = datafiles[3]

testImages = datafiles[0]
testLabels = datafiles[1]

numFilters1 = 80
numFilters2 = 50
numFilters3 = 25
numFilters4 = 16

kernelSize1 = 5
kernelSize2 = 5
kernelSize3 = 1
kernelSize4 = 1

num_channels = 1
learningRate = 0.01
nFold = 5 ##N-Fold Cross-Validation
batchSize = 32

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

savePath = '/Users/Jon/Documents/Workspace/Machine_Learning/research/params/conv.ckpt'

'''
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel '''

# with open(datafiles[0], 'rb') as f:
#     f.seek(4, 1)
#     num_images = int.from_bytes(f.read(4), byteorder='big')
#     num_rows = int.from_bytes(f.read(4), byteorder='big')
#     num_cols = int.from_bytes(f.read(4), byteorder='big')
#     print(num_images, num_rows, num_cols)
#     f.close()

def createSet():
    trainingSet = []
    testingSet = []
    with open(trainImages, 'rb') as f:
        f.seek(4, 1)
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')
        num_pixels = num_rows * num_cols
        for i in range(0, num_images):
            rawPic = f.read(num_pixels)
            imgdata = np.frombuffer(rawPic, dtype=np.uint8,count=num_pixels)
            img = np.reshape(imgdata, (num_rows, num_cols, num_channels))
            dict = {'image':img, 'label':None, 'id':i}
            trainingSet.append(dict)

    with open(trainLabels, 'rb') as f:
        f.seek(4, 1)
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        for i in range(0, num_labels):
            rawLabel = f.read(1)
            l = int.from_bytes(rawLabel, byteorder='big')
            # labeldata = np.frombuffer(rawLabel, dtype=np.uint8, count=1)
            labeldata = np.zeros((1, 10), dtype=float)
            labeldata[0][l] = 1
            trainingSet[i]['label'] = labeldata

    with open(testImages, 'rb') as f:
        with open(trainImages, 'rb') as f:
            f.seek(4, 1)
            num_images = int.from_bytes(f.read(4), byteorder='big')
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')
            num_pixels = num_rows * num_cols
            for i in range(0, num_images):
                rawPic = f.read(num_pixels)
                imgdata = np.frombuffer(rawPic, dtype=np.uint8, count=num_pixels)
                img = np.reshape(imgdata, (num_rows, num_cols, num_channels))
                dict = {'image': img, 'label': None, 'id': i}
                testingSet.append(dict)

    with open(testLabels, 'rb') as f:
        f.seek(4, 1)
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        for i in range(0, num_labels):
            rawLabel = f.read(1)
            l = int.from_bytes(rawLabel, byteorder='big')
            # labeldata = np.frombuffer(rawLabel, dtype=np.uint8, count=1)
            labeldata = np.zeros((1, 10), dtype=float)
            labeldata[0][l] = 1
            testingSet[i]['label'] = labeldata
    random.shuffle(testingSet)
    random.shuffle(trainingSet)
    sets = {'testing':testingSet, 'training':trainingSet}

    return sets

def model(checkpoint=False):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, num_channels), name='inputs')
        labels = tf.placeholder(dtype=tf.float32, shape=(None, 1, 10), name='labels')

        filters1 = tf.Variable(tf.random_normal((kernelSize1, kernelSize1, 1, numFilters1), stddev=0.1))
        conv1 = tf.nn.convolution(inputs, filters1, padding='SAME', data_format='NHWC')
        relconv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.pool(relconv1, window_shape=[2, 2], pooling_type='MAX', padding='VALID', strides=[2, 2],
                           data_format='NHWC')

        filters2 = tf.Variable(tf.random_normal((kernelSize2, kernelSize2, numFilters1, numFilters2), stddev=0.1))
        conv2 = tf.nn.convolution(pool1, filters2, padding='SAME', data_format='NHWC')
        relconv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.pool(relconv2, window_shape=[2, 2], pooling_type='MAX', padding='VALID', strides=[2, 2],
                           data_format='NHWC')

        filters3 = tf.Variable(tf.random_normal((kernelSize3, kernelSize3, numFilters2, numFilters3), stddev=0.1))
        conv3 = tf.nn.convolution(pool2, filters3, padding='SAME', data_format='NHWC')
        relconv3 = tf.nn.relu(conv3)

        filters4 = tf.Variable(tf.random_normal((kernelSize4, kernelSize4, numFilters3, numFilters4), stddev=0.1))
        conv4 = tf.nn.convolution(relconv3, filters4, padding='SAME', data_format='NHWC')
        relconv4 = tf.nn.relu(conv4)
        shape = tf.shape(relconv4)

        # denseLayerInputs = tf.reshape(conv4, shape=[1, 392])
        denseLayerInputs = tf.contrib.layers.flatten(relconv4)
        denseWeights1 = tf.Variable(tf.random_normal((784, 50), stddev=0.1))
        denseBias1 = tf.Variable(tf.random_normal((1, 50), stddev=0.1))
        hiddenActivation1 = tf.nn.relu(tf.add(tf.matmul(denseLayerInputs, denseWeights1), denseBias1))

        denseWeights2 = tf.Variable(tf.random_normal((50, 20), stddev=0.1))
        denseBias2 = tf.Variable(tf.random_normal((1, 20)))
        hiddenActivation2 = tf.nn.relu(tf.add(tf.matmul(hiddenActivation1, denseWeights2), denseBias2))

        denseWeights3 = tf.Variable(tf.random_normal((20, 10), stddev=0.1))
        denseBias3 = tf.Variable(tf.random_normal((1, 10), stddev=0.1))
        hiddenActivation3 = tf.nn.relu(tf.add(tf.matmul(hiddenActivation2, denseWeights3), denseBias3))

        denseWeights4 = tf.Variable(tf.random_normal((10, 10), stddev=0.1))
        denseBias4 = tf.Variable(tf.random_normal((1, 10), stddev=0.1))
        hiddenActivation4 = tf.nn.relu(tf.add(tf.matmul(hiddenActivation3, denseWeights4), denseBias4))

        output = tf.nn.softmax(hiddenActivation4)

        finalOutput = tf.squeeze(output)
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(labels, finalOutput))
        train = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

        maxIndices = tf.argmax(finalOutput, axis=1)
        predictions = tf.one_hot(maxIndices, depth=10, on_value=1, off_value=0, axis=1, dtype=tf.int32)
        batchCorrect = tf.equal(tf.argmax(tf.cast(labels, dtype=tf.int32), axis=2), tf.argmax(predictions, axis=1))
        batchAccuracy = tf.reduce_mean(tf.cast(batchCorrect, dtype=tf.float32))

        runningAccuracy, accuracyUpdate = tf.metrics.accuracy(tf.argmax(tf.cast(labels, dtype=tf.int32), axis=2), tf.argmax(predictions, axis=1))
        init = tf.global_variables_initializer()
        linit = tf.local_variables_initializer()

    operations = {'debug':shape,
                  'init':init,
                  'inputs':inputs,
                  'labels':labels,
                  'output':output,
                  'train':train,
                  'predictions':predictions,
                  'batchCorrect':batchCorrect,
                  'batchAccuracy':batchAccuracy,
                  'runningAccuracy':runningAccuracy,
                  'accuracyUpdate':accuracyUpdate,
                  'linit':linit
                  }
    return operations, graph

def partitionSets(n, sets):
    '''partitions sets for n fold cross validation'''
    trainingSet = sets['training']
    assert n < len(trainingSet)
    partitions = [[]]*5
    for i in range(0, len(trainingSet)):
        partitions[i%n].append(trainingSet[i])
    return partitions

def createFeedDict(partitions, validationPartition, ops, batchSize, n, shuffle=False, runningCounts=None):
    activePartitions = []
    end = False
    if shuffle:
        for p in partitions:
            random.shuffle(p)
    for i in range(0, n):
        if i != validationPartition:
            activePartitions.append(partitions[i][:])
    batchSet = []
    counts = [0]*(n-1)
    if runningCounts != None:
        counts = runningCounts
    num_sampled = sum(counts)
    partition_entries = sum([len(x) for x in activePartitions])
    diff = partition_entries-num_sampled
    if (diff) < batchSize:
        batchSize = diff
        end=True
        ##make sure not to grab more data than is there
    for i in range(0, batchSize):
        part = i % (n - 1)
        batchSet.append(activePartitions[part][counts[part]])
        counts[part] += 1
    imageslist = []
    labelslist = []
    for item in batchSet:
        imageslist.append(item['image'])
        labelslist.append(item['label'])
    images = np.array(imageslist).reshape((len(imageslist), 28, 28, 1))
    labels = np.array(labelslist).reshape((len(labelslist), 1, 10))

    fdict = {ops['inputs']:images, ops['labels']:labels}
    return fdict, counts, end

def validationDict(partitions, validationPartition, ops):
    imageslist = []
    labelslist = []
    for item in partitions[validationPartition]:
        imageslist.append(item['image'])
        labelslist.append(item['label'])
    images = np.array(imageslist).reshape((len(imageslist), 28, 28, 1))
    labels = np.array(labelslist).reshape((len(labelslist), 1, 10))
    fdict = {ops['inputs']: images, ops['labels']: labels}
    return fdict

def testDict(fullset, ops):
    imageslist = []
    labelslist = []
    for items in fullset['testing']:
        imageslist.append(items['image'])
        labelslist.append(items['labels'])
    images = np.array(imageslist).reshape((len(imageslist), 28, 28, 1))
    labels = np.array(labelslist).reshape((len(labelslist), 1, 10))
    fdict = {ops['inputs']: images, ops['labels']: labels}
    return fdict

'''
model should execute an epoch, compute the accuracy of the validation partition,
and if the validation error goes up, the network stops training
'''

ops, graph = model()
checkpointLoad = False
if(len(sys.argv)) != 1:
    if sys.argv[1] == '-c':
        checkpointLoad = True

with tf.Session(config=config, graph=graph) as sess:
    fullset = createSet()
    partitions = partitionSets(5, fullset)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver()
    validationAccuracy = 0
    partitionIndicator = 0

    sess.run([ops['init'], ops['linit']])
    if checkpointLoad:
        saver.restore(sess, savePath)
        print("Restoring old model...")

    def handler(signum, frame):
        print("\nmodel saving before sigkill {}".format(signum))
        saver.save(sess, save_path=savePath)
        print("Have a nice day!")
        exit(0)


    signal.signal(signal.SIGINT, handler=handler)
    # make it such that training can be stopped at any time
    # handler is defined here so it can use the saver

    while not coord.should_stop():
        counts = [0] * (nFold - 1)
        first = True
        end = False
        while True:

            fdict, counts, end = createFeedDict(partitions, partitionIndicator, ops, batchSize, nFold, first,
                                                counts)
            num_processed = sum(counts)
            first = False
            oplist = [
                ops['batchAccuracy'],
                ops['train'],
                ops['accuracyUpdate'],
                ops['predictions'],
                ops['labels'],
                ops['output'],
                ops['debug']
            ]
            batchAcc, _g, runningAcc, predictions, labels, out, debug = sess.run(oplist, feed_dict=fdict)
            # print("Batch Accuracy is at {}%".format(batchAcc))
            print("Running Accuracy is {}%".format(runningAcc))
            print("processed: {}/60000 ({}%)".format(num_processed, num_processed/60000))
            # print("debug: {}".format(debug))
            # print("Softmax outputs: {}".format(out))
            print("Predictions are: {}".format(predictions.argmax(axis=1)))
            labels = labels.reshape((labels.shape[0], 10))
            print("Labels are:      {}".format(labels.argmax(axis=1)))
            if end:
                break
        # now perform the validation step
        vfdict = validationDict(partitions, partitionIndicator, ops)
        vbatchAcc = sess.run([ops['batchAccuracy']], feed_dict=vfdict)
        if vbatchAcc < validationAccuracy:
            print("Training will now stop")
            saver.save(sess, save_path=savePath)
            coord.should_stop()
            tdict = testDict(fullset, ops)
            testAccuracy = sess.run(ops['batchAccuracy'], feed_dict=tdict)
            print("Testing accuracy is {}".format(testAccuracy))
        else:
            validationAccuracy = vbatchAcc
        partitionIndicator += 1
        partitionIndicator %= nFold

    coord.join()
    saver.save(sess, save_path=savePath)
    print("Have a nice day!")

