# mushroom-dataset-analysis
A classifier program that trains a model to distinguish edible from poisonous mushrooms from the mushrooms dataset using a PyTorch neural network or a sklearn decision tree.
The dataset contains data from 8124 mushrooms.
Each record is a set of categorical features describing physical attributes of the mushroom. The features are one hot encoded using the pandas library. K-fold cross validation is employed, coded by hand.
Metrics used for evaluation are runtime, precision and recall.
Additionally, NN training can be run on the user's GPU for performance improvement.

Part of a coursework for 403 Data Mining at Lancaster University.

DATASET: https://archive.ics.uci.edu/ml/datasets/mushroom
