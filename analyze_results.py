#!/usr/bin/python3
import pandas as pd
import numpy as np
import glob
import os, sys
import scipy.stats as stat

# Calculate accuracy percentage between two lists
def get_accuracy(actual, predicted):
        ncorrect = 0
        for i in range(len(actual)):
                if actual[i] == predicted[i]:
                        ncorrect += 1
        return ncorrect / float(len(actual)) * 100.0


# calculate a confusion matrix
def confusion_matrix(actual, predicted):
        unique = set(np.sort(actual))
        matrix = [list() for x in range(len(unique))]
        for i in range(len(unique)):
                matrix[i] = [0 for x in range(len(unique))]
        lookup = dict()
        for i, value in enumerate(unique):
                lookup[value] = i
        for i in range(len(actual)):
                x = lookup[actual[i]]
                y = lookup[predicted[i]]
                matrix[x][y] += 1
        return unique, matrix


# print a confusion matrix
def print_confusion_matrix(unique, matrix):
        print('(P)    ' + '  '.join(str(x) for x in unique))
        #print('(A)**')
        for i, x in enumerate(unique):
                print("(A) %s| %s" % (x, '  '.join(str(x) for x in matrix[i])))




def main(argv):
        actual_labels_fname = sys.argv[1]
        predicted_labels_fname = sys.argv[2]
        print("actual labels: %s\n" % actual_labels_fname);
        print("predicted labels: %s\n" % predicted_labels_fname);

        with open(actual_labels_fname) as f:
            actual = f.readlines()
        f.close();
        with open(predicted_labels_fname) as f:
            predicted = f.readlines()
        f.close();

        actual = [ int(x) for x in actual ] 
        predicted = [ int(x) for x in predicted ] 
        print(">>>>>>>> actual label: %d\n" % actual[0]);
        print(">>>>>>>> predicted label: %d\n" % predicted[0]);

        acc = get_accuracy(actual,predicted);
        print(">>>>>>>> classification accuracy: %4.2f\n" % acc);

        print("confusion matrix:\n");
        unique,matrix=confusion_matrix(actual, predicted);
        #print(unique);
        print_confusion_matrix(unique, matrix);

if __name__ == "__main__":
	main(sys.argv[1:])

