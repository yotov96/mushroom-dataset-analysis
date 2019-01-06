import numpy as np
import pandas as pd
import torch
import time
from sklearn.tree import DecisionTreeClassifier


# method to split data into chunks for k_fold cross-validation
def data_split(data, labels, sector, n_chunks):
    sector_size = int(data.shape[0] / n_chunks)  # split dataset into chunks for cross validation
    if sector == n_chunks - 1:
        test_sector = data[sector * sector_size:, :]
        test_labels = labels[sector * sector_size:, :]
    else:
        test_sector = data[sector * sector_size: (sector + 1) * sector_size, :]
        test_labels = labels[sector * sector_size: (sector + 1) * sector_size, :]
    if sector > 0:
        training_sector1 = data[: sector * sector_size, :]
        training_sector2 = data[(sector + 1) * sector_size:, :]
        training_sectors = np.append(training_sector1, training_sector2, 0)
        training_labels1 = labels[: sector * sector_size, :]
        training_labels2 = labels[(sector + 1) * sector_size:, :]
        training_labels = np.append(training_labels1, training_labels2, 0)
    else:
        training_sectors = data[(sector + 1) * sector_size:, :]
        training_labels = labels[(sector + 1) * sector_size:, :]
    return training_sectors, training_labels, test_sector, test_labels


def NN_train(training_data, training_labels, gpu = False):
    labels = torch.tensor(training_labels, dtype = torch.float32)  # transform training labels to torch tensors
    data = torch.from_numpy(training_data)  # transform training data to torch tensors

    # create sequential neural network, first layer is the number of features, hidden layer is features/2,
    # output layer is 1, with applying logistic regression between layers
    model = torch.nn.Sequential(torch.nn.Linear(training_data.shape[1], int(training_data.shape[1] / 2)),
                                torch.nn.Sigmoid(), torch.nn.Linear(int(training_data.shape[1] / 2), 1),
                                torch.nn.Sigmoid())
    loss_fn = torch.nn.MSELoss(reduction = 'sum')  # choosing loss function
    # if running on gpu set gpu flags using cuda()
    if gpu:
        labels = labels.cuda()
        data = data.cuda()
        loss_fn = loss_fn.cuda()
        model = torch.nn.Sequential(torch.nn.Linear(training_data.shape[1], int(training_data.shape[1] / 2)).cuda(),
            torch.nn.Sigmoid().cuda(), torch.nn.Linear(int(training_data.shape[1] / 2), 1).cuda(),
            torch.nn.Sigmoid().cuda())
        model = model.cuda()
    step = 0.001
    n_epochs = 1000
    for t in range(n_epochs):
        training_prediction = model(data)  # prediction
        loss = loss_fn(training_prediction, labels)  # calculate loss
        model.zero_grad()  # zero the gradient
        loss.backward()  # back propagation
        with torch.no_grad():
            for param in model.parameters():
                param -= step * param.grad
    model.eval()  # now evaluating the model
    return model



###################################
# load mushrooms data from file
data_file = open("agaricus-lepiota.data", "r")
# two lists to hold data
missing_data = []
full_data = []
# read through file
while True:
    line = data_file.readline().rstrip("\n")
    if len(line) == 0:
        break
    read_data = line.split(",")
    full_data.append(read_data)

full_data = np.array(full_data)
###################################
#columns_to_delete = [11] #full dataset without missing values
#columns_to_delete = [] #full dataset with missing values
#columns_to_delete = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]  # odor, habitat, spore print color
#columns_to_delete = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]  # cap color,habitat, spore print color
#columns_to_delete = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]      # cap color, odor, habitat, spore print color
columns_to_delete = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]      # cap color, odor, habitat, population, spore print color
full_data = np.delete(full_data, columns_to_delete, 1)  # delete unwanted features

columns = full_data.shape[1]

full_data_encoded = np.zeros([8124, 1])                         # numpy array for encoded date
for column_index in range(columns):                             # using one hot encoding for categorical features
    full_data_encoded = np.append(full_data_encoded, pd.get_dummies(full_data[:, column_index], 1), 1)  # one hot encoding

raw_labels = full_data_encoded[:, [2]]                          # extract labels
full_data_encoded = np.delete(full_data_encoded, [0, 1, 2], 1)  # delete labels and initial zeros column from dataset

full_data_encoded = full_data_encoded.astype(np.float32)
raw_labels = raw_labels.astype(np.float32)

torch.cuda.set_device(0)    # for using GPU with torch
n_chunks = 10               # number of sections for cross-validation i.e how many chunks to split data into
current_round = 0
# variables to hold results data
truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0
correct = 0
total = 0
mode = input("Enter 'n' for NN and anything else for decision tree classifier: ")
if mode == 'n':
    gpumode = input("Enter 'y' to run NN training on GPU: ")
    if gpumode == 'y':
        gpu = True
    else:
        gpu = False
start = time.time()
while current_round < n_chunks:
    training_data, training_labels, test_data, test_labels = data_split(full_data_encoded, raw_labels, current_round, n_chunks)  # splitter function
    if mode == 'n':
        model = NN_train(training_data, training_labels, gpu)
        for item_index in range(test_data.shape[0]):
            with torch.no_grad():
                item = torch.from_numpy(test_data[item_index])
                if gpu:
                    item = item.cuda()
                prediction = model(item)  # get the prediction
                print("Item: " + str(item_index))
                print("\nNN prediction: " + str(prediction.item()))
                print("Item actual: " + str(int(test_labels[item_index])))
                if prediction.item() > 0.5 and int(test_labels[item_index]) == 1:
                    truePositive += 1
                    correct += 1
                elif prediction.item() < 0.5 and int(test_labels[item_index]) == 0:
                    trueNegative += 1
                    correct += 1
                elif prediction.item() < 0.5 and int(test_labels[item_index]) == 1:
                    falseNegative += 1
                elif prediction.item() > 0.5 and int(test_labels[item_index]) == 0:
                    falsePositive += 1
            total += 1
    else:
        # create decision tree using gini index for purity calculation
        decision_tree = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 4, min_samples_leaf = 3)
        decision_tree.fit(training_data, training_labels)  # train the model
        predictions = decision_tree.predict(test_data)     # predict on the test data
        for index in range(len(predictions)):
            print("\nDecision tree prediction: " + str(predictions[index]))
            print("Item actual: " + str(int(test_labels[index])))
            if predictions[index] == 1 and test_labels[index] == 1:
                correct += 1
                truePositive += 1
            elif predictions[index] == 0 and test_labels[index] == 0:
                trueNegative += 1
                correct += 1
            elif predictions[index] == 0 and test_labels[index] == 1:
                falseNegative += 1
            elif predictions[index] == 1 and test_labels[index] == 0:
                falsePositive += 1
            total += 1
    current_round += 1

print('\nAccuracy: ' + str(correct) + '/' + str(total))
print('True positive: ' + str(truePositive) + '/' + str(3916))
print('True negative: ' + str(trueNegative) + '/' + str(4208))
print('False positive: ' + str(falsePositive))
print('False negative: ' + str(falseNegative))
print('Precision: ' + str(truePositive/(truePositive + falsePositive)))
print('Recall: ' + str(truePositive/(truePositive + falseNegative)))
end = time.time()
print("Code execution time: " + str(end - start) + " seconds")