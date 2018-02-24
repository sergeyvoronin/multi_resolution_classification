from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from numpy import array
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import sklearn
import sklearn.metrics
import numpy as np
import os
import csv

# run LSTM multiple sequences (1+8 feature sets) to one classification problem
# Sergey Voronin

def printf(format, *values):
    print(format % values)


def print_detection_rate(y_test, y_pred):
    inds_ones_y_test = np.where(y_test == 1)
    inds_ones_y_test = inds_ones_y_test[0]
    inds_ones_y_pred = np.where(y_pred == 1)
    inds_ones_y_pred = inds_ones_y_pred[0]
    score = float(len(np.intersect1d(inds_ones_y_test,
                                     inds_ones_y_pred))) / float(len(inds_ones_y_test)) * 100
    printf("detection rate = %4.4f\n", score)
    return


def print_falsepositive_rate(y_test, y_pred):
    inds_ones_y_test = np.where(y_test == 1)
    inds_ones_y_test = inds_ones_y_test[0]
    inds_ones_y_pred = np.where(y_pred == 1)
    inds_ones_y_pred = inds_ones_y_pred[0]
    num_fp = 0
    for ind in inds_ones_y_pred:
        if y_test[ind] == 0:
            num_fp = num_fp + 1
    score = float(num_fp) / float(len(inds_ones_y_pred)) * 100
    printf("falsepositives rate = %4.4f\n", score)
    return


def get_training_data2(num_files):
    X = list()
    y = list()

    # ut
    unt_rows = list()
    with open('data/train_samples_unt.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            unt_rows.append(row)
            ind = ind + 1

    # wav1 app
    wav1ap_rows = list()
    with open('data/train_samples_wavelet1ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav1ap_rows.append(row)
            ind = ind + 1

    # wave1 dl
    wav1dl_rows = list()
    with open('data/train_samples_wavelet1dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav1dl_rows.append(row)
            ind = ind + 1



def get_training_data(num_files):
    X = list()
    y = list()

    # unt
    unt_rows = []
    with open('data/train_samples_unt.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            unt_rows.insert(ind,row)
            ind = ind + 1

    # wav1 app
    wav1ap_rows = [] 
    with open('data/train_samples_wavelet1ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav1ap_rows.insert(ind,row)
            ind = ind + 1

    # wave1 dl
    wav1dl_rows = [] 
    with open('data/train_samples_wavelet1dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav1dl_rows.insert(ind,row)
            ind = ind + 1

    # wav2 app
    wav2ap_rows = [] 
    with open('data/train_samples_wavelet2ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav2ap_rows.insert(ind,row)
            ind = ind + 1

    # wav2 dl
    wav2dl_rows = [] 
    with open('data/train_samples_wavelet2dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav2dl_rows.insert(ind,row)
            ind = ind + 1


    # wav3 app
    wav3ap_rows = [] 
    with open('data/train_samples_wavelet3ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav3ap_rows.insert(ind,row)
            ind = ind + 1

    # wav3 dl
    wav3dl_rows = [] 
    with open('data/train_samples_wavelet3dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav3dl_rows.insert(ind,row)
            ind = ind + 1

	
    # wav4 app
    wav4ap_rows = [] 
    with open('data/train_samples_wavelet3ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav4ap_rows.insert(ind,row)
            ind = ind + 1

    # wav4 dl
    wav4dl_rows = [] 
    with open('data/train_samples_wavelet3dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav4dl_rows.insert(ind,row)
            ind = ind + 1

    for i in range(1,num_files+1):
        seq = [np.array(unt_rows[i]), np.array(wav1ap_rows[i]), np.array(wav1dl_rows[i]), np.array(wav2ap_rows[i]), np.array(wav2dl_rows[i]), np.array(wav3ap_rows[i]), np.array(wav3dl_rows[i]), np.array(wav4ap_rows[i]), np.array(wav4dl_rows[i])]
        X.append(seq)
        
    with open('data/train_labels1.txt') as f:
        for line in f:
            y.append(int(line));

    X = array(X);
    y = array(y);
    print(X.shape);
    print(X[1][0]);

    #X = X[0];
    #Xnew = np.zeros((num_files,9,59-1));
    Xnew = np.zeros((num_files,9,54-1));
    for i in range(0,num_files):
        Xnew[i][0] = X[i][0]
        Xnew[i][1][0:41] = X[i][1]
        Xnew[i][2][0:41] = X[i][2]
        Xnew[i][3][0:41] = X[i][3]
        Xnew[i][4][0:41] = X[i][4]
        Xnew[i][5][0:41] = X[i][5]
        Xnew[i][6][0:41] = X[i][6]
        Xnew[i][7][0:41] = X[i][7]
        Xnew[i][8][0:41] = X[i][8]

    return Xnew, y



def get_testing_data(num_files):
    X = list()
    y = list()

    # unt
    unt_rows = []
    with open('data/test_samples_unt.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            unt_rows.insert(ind,row)
            ind = ind + 1

    # wav1 app
    wav1ap_rows = [] 
    with open('data/test_samples_wavelet1ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav1ap_rows.insert(ind,row)
            ind = ind + 1

    # wav1 dl
    wav1dl_rows = [] 
    with open('data/test_samples_wavelet1dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav1dl_rows.insert(ind,row)
            ind = ind + 1


    # wav2 app
    wav2ap_rows = [] 
    with open('data/test_samples_wavelet2ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav2ap_rows.insert(ind,row)
            ind = ind + 1

    # wav2 dl
    wav2dl_rows = [] 
    with open('data/test_samples_wavelet2dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav2dl_rows.insert(ind,row)
            ind = ind + 1


    # wav3 app
    wav3ap_rows = [] 
    with open('data/test_samples_wavelet3ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav3ap_rows.insert(ind,row)
            ind = ind + 1

    # wav3 dl
    wav3dl_rows = [] 
    with open('data/test_samples_wavelet3dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav3dl_rows.insert(ind,row)
            ind = ind + 1

    # wav4 app
    wav4ap_rows = [] 
    with open('data/test_samples_wavelet4ap.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav4ap_rows.insert(ind,row)
            ind = ind + 1

    # wav4 dl
    wav4dl_rows = [] 
    with open('data/test_samples_wavelet4dl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        ind = 0
        for row in readCSV:
            del row[0]
            if(ind > 0):
                row = [float(i) for i in row]
            wav4dl_rows.insert(ind,row)
            ind = ind + 1

    for i in range(1,num_files+1):
        seq = [np.array(unt_rows[i]), np.array(wav1ap_rows[i]), np.array(wav1dl_rows[i]), np.array(wav2ap_rows[i]), np.array(wav2dl_rows[i]), np.array(wav3ap_rows[i]), np.array(wav3dl_rows[i]), np.array(wav4ap_rows[i]), np.array(wav4dl_rows[i])]
        X.append(seq)
        
    with open('data/test_labels1.txt') as f:
        for line in f:
            y.append(int(line));

    X = array(X);
    y = array(y);

    #X = X[0];
    #Xnew = np.zeros((1,9,59-6));
#    Xnew = np.zeros((num_files,9,54-1));
#    for i in range(0,num_files):
#        Xnew[i][0] = X[i][0]
#        Xnew[i][1][0:41] = X[i][1]
#        Xnew[i][2][0:41] = X[i][2]
#        Xnew[i][3][0:41] = X[i][3]
#        Xnew[i][4][0:41] = X[i][4]
#        Xnew[i][5][0:41] = X[i][5]
#        Xnew[i][6][0:41] = X[i][6]
#        Xnew[i][7][0:41] = X[i][7]
#        Xnew[i][8][0:41] = X[i][8]
    Xnew = np.zeros((num_files,9,54-1));
    for i in range(0,num_files):
        Xnew[i][0] = X[i][0]
        Xnew[i][1][0:41] = X[i][1]
        Xnew[i][2][0:41] = X[i][2]
        Xnew[i][3][0:41] = X[i][3]
        Xnew[i][4][0:41] = X[i][4]
        Xnew[i][5][0:41] = X[i][5]
        Xnew[i][6][0:41] = X[i][6]
        Xnew[i][7][0:41] = X[i][7]
        Xnew[i][8][0:41] = X[i][8]

    return Xnew, y



# get train data
print("loading data\n");
num_files = 319 + 80;
print("getting train data\n");
Xtr, ytr = get_training_data(319)
print("getting test data\n");
Xtest, ytest = get_testing_data(80)

ytr_cat = to_categorical(ytr-1, num_classes=10)
ytest_cat = to_categorical(ytest-1, num_classes=10)


# define a stacked LSTM model with several hidden layers
print("building model\n");
model = Sequential()
model.add(LSTM(80, return_sequences=True, input_shape=(9, 54-1)))
model.add(LSTM(80, return_sequences=True))
model.add(LSTM(80))
#model.add(Activation('relu'))
## model.add(Dense(1, activation='linear')) # use this for speed up
#model.add(Dense(1,activation='sigmoid'))
model.add(Dense(10,activation='softmax'))
#
## compile model
#model.compile(loss='mse', optimizer='adam',  class_mode='binary')
print("fitting model\n");
#model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#
## fit model
##model.fit(Xtr, ytr, batch_size = 5, epochs=200, shuffle=False)
model.fit(Xtr, ytr_cat, epochs=200, shuffle=False)
#model.fit(Xtr, ytr_cat, epochs=30, shuffle=False)
#
## make predictions
predictions = []
actuals = []
yhat = model.predict(Xtest, verbose=0)
for i in range(0,len(yhat)):
    max_ind = np.argmax(yhat[i]) + 1;
    predictions.append(max_ind);
    printf("prediction = %d\n", max_ind)
    printf("actual = %d\n", ytest[i])

file = open("data/output_lstm.txt","w") 
for i in range(0,len(yhat)):
    file.write("%d\n" % (np.argmax(yhat[i]) + 1));
file.close();

# output results
predictions=  array(predictions);

# per-class results output
report =  sklearn.metrics.classification_report(ytest,array(predictions))
print(report)

