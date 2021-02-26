"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC

from align.DetectFace import DetectFace
import cv2


class Classifier:
    def __init__(self, args):
        with tf.compat.v1.Graph().as_default():
            self.sess = tf.compat.v1.Session()

            np.random.seed(seed=args.seed)

            self.batch_size = args.batch_size
            self.image_size = args.image_size_
            self.use_split_dataset = args.use_split_dataset
            self.min_nrof_images_per_class = args.min_nrof_images_per_class
            self.nrof_train_images_per_class = args.nrof_train_images_per_class
            self.mode = args.mode

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]

            self.classifier_filename_exp = os.path.expanduser(args.classifier_filename)

    def splitDataset(self, dataset):
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            # Remove classes with less than min_nrof_images_per_class
            if len(paths) >= self.min_nrof_images_per_class:
                np.random.shuffle(paths)
                train_set.append(facenet.ImageClass(cls.name, paths[:self.nrof_train_images_per_class]))
                test_set.append(facenet.ImageClass(cls.name, paths[self.nrof_train_images_per_class:]))
        return train_set, test_set

    def trainImages(self, data_dir, use_dataset):
        if self.use_split_dataset:
            dataset_tmp = facenet.get_dataset(data_dir)
            train_set, test_set = self.splitDataset(dataset_tmp)
            if self.mode == 'TRAIN':
                dataset = train_set
            elif self.mode == 'CLASSIFY':
                dataset = test_set
        else:
            dataset = facenet.get_dataset(data_dir)

        # Check that there are at least one training image per class
        for cls in dataset:
            assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

        paths, labels = facenet.get_image_paths_and_labels(dataset)

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, self.image_size)
            feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
        # Train classifier
        print('Training classifier')
        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(self.classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % self.classifier_filename_exp)

    def classifyImages(self, data_dir, use_dataset):
        dataset = []
        if use_dataset is True:
            if self.use_split_dataset:
                dataset_tmp = facenet.get_dataset(data_dir)
                train_set, test_set = self.splitDataset(dataset_tmp)
                if self.mode == 'TRAIN':
                    dataset = train_set
                elif self.mode == 'CLASSIFY':
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)
        else:
            for face in data_dir:
                dataset.append([face, 'unknown'])
            paths, labels = facenet.get_image_paths_and_labels(dataset)

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, self.image_size)
            feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
        # Classify images
        print('Testing classifier')
        with open(self.classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % self.classifier_filename_exp)

        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        accuracy = np.mean(np.equal(best_class_indices, labels))
        print('Accuracy: %.3f' % accuracy)


def main(args):
    cap = cv2.VideoCapture(0)
    classifier = Classifier(args)
    detectFace = DetectFace(args)
    while True:
        ret, img = cap.read()
        det_faces, res_img = detectFace.getFaces(img)
        cv2.imshow('detectedFaces', res_img)
        cv2.waitKey(1)
        if len(det_faces) > 0:
            if args.mode == 'TRAIN':
                classifier.trainImages(det_faces, use_dataset=True)
            elif args.mode == 'CLASSIFY':
                classifier.classifyImages(det_faces, use_dataset=True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Parameters for detecting faces using mtcnn
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)

    # Parameters for identifying faces using facenet
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size_', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
