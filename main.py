import scipy.io as spio
import numpy as np
from collections import Counter
import sys
import matplotlib.pyplot as plt

numEpochs = 100

def count_labels(yTrain):
    
    print(Counter(yTrain.flatten()))


def normalize(target_array):
    
    return target_array * (2 / 255) - 1


def one_hot(target_array):

    a = target_array[0]
    b = np.zeros((10, a.size))
    b[a, np.arange(a.size)] = 1
    return b


def forward_pass(x, y, W1, W2, b1, b2):

    m = x.shape[1]

    z1 = np.dot(W1, x) + b1

    vec_relu = np.vectorize(relu)
    h1 = vec_relu(z1)

    z2 = np.dot(W2, h1) + b2
    y_hat = np.apply_along_axis(softmax, 0, z2)

    return [z1, z2, h1, y_hat]


def softmax(x):

    a = x.max()
    b = x - a

    exp_x = np.exp(b)
    res = exp_x / np.sum(exp_x)
    return res


def relu(x):
    
    return max(x, 0)


def cross_entropy_loss(y, y_hat):

    samples = y.shape[1]
    return (-1/ samples) * np.sum(y * np.log(y_hat))


def accuracy(y, y_hat):

    x = 0
    samples = y.shape[1]

    yT = y.T
    y_hatT = y_hat.T
    
    for i in range(0, samples):
        max_i = np.argmax(y_hatT[i])

        if yT[i][max_i] == 1:
            x += 1
    
    return x / samples * 100


def partial_relu(x):

    return 1 if x > 0 else 0


def backward_pass(x, y, y_hat, z1, h1, W1, W2, b1, b2, learn_rate):

    alpha = learn_rate
    m = x.shape[1]

    delta2 = (y_hat - y) / m

    deltaLW2 = np.dot(delta2, h1.T)

    deltaLb2 = np.sum(delta2, axis=1).reshape(delta2.shape[0], 1)

    vec_relu = np.vectorize(partial_relu)
    pRelu = vec_relu(z1)
    delta1 = np.dot(W2.T, delta2) * pRelu
    
    deltaLW1 = np.dot(delta1, x.T)
    deltaLb1 = np.sum(delta1, axis=1).reshape(delta1.shape[0], 1)
    
    newW1 = W1 - alpha * deltaLW1
    newW2 = W2 - alpha * deltaLW2

    newb1 = b1 - alpha * deltaLb1
    newb2 = b2 - alpha * deltaLb2

    return [deltaLW1, deltaLb1, deltaLW2, deltaLb2, newW1, newb1, newW2, newb2]


def main():
    np.random.seed(0)
    data = spio.loadmat('mnistReduced.mat')

    xTrain = data['images_train']  # 784*30000
    yTrain = data['labels_train']  # 1*30000

    xVal = data['images_val']  # 784*3000
    yVal = data['labels_val'] # 1*3000

    xTest = data['images_test'] # 784*3000
    yTest = data['labels_test'] # 1*3000
    # proceed to normalize xTrain, xVal, xTest and then converting yTrain, yVal, yTest to one-hot encoding

    # Count Labels from the training set
    # count_labels(yTrain)

    # Normalize Values from [0, 255] to [-1, 1]
    xTrain = normalize(xTrain)
    xVal = normalize(xVal)
    xTest = normalize(xTest)
    
    # One-Hot-Encoding for Labels
    yTrain = one_hot(yTrain)
    yVal = one_hot(yVal)
    yTest = one_hot(yTest)

    samples = xTrain.shape[1]
    batchSize = 256
    learning_rates = [0.001, 0.01, 0.1, 1, 10]

    test_results = {
        'loss' : [],
        'acc' : []
    }
    xTrainSplit = []
    yTrainSplit = []

    end = False
    for batchStart in range(0, samples + batchSize, batchSize):
        batchEnd = batchStart + batchSize
        if batchEnd > samples:
            end = True
            batchEnd = samples
        xTrainSplit.append(xTrain[:, batchStart:batchEnd])
        yTrainSplit.append(yTrain[:, batchStart:batchEnd])
        if end:
            break
    
    learn_rate = 0.1
    W1 = 0.0001 * np.random.randn(30, 784)
    W2 = 0.0001 * np.random.randn(10, 30)
    b1 = np.zeros((30, 1))
    b2 = np.zeros((10, 1))
    results = {
        'training' : {
            'loss' : [],
            'acc' : []
        },
        'validation' : {
            'loss' : [],
            'acc' : []
        }
    }
    last_epoch = 0
    in_a_row = 0
    for epoch in range(0, numEpochs):
        print("In Epoch {}".format(epoch))
        for split in range(0, 118):

            [z1, z2, h1, yhat] = forward_pass(
                xTrainSplit[split],
                yTrainSplit[split],
                W1, W2, b1, b2
            )

            [gradW1, gradb1, gradW2, gradb2, newW1, newb1, newW2, newb2] = backward_pass(
                xTrainSplit[split],
                yTrainSplit[split],
                yhat, z1, h1, W1, W2, b1, b2, learn_rate
            )

            W1 = newW1
            W2 = newW2
            b1 = newb1
            b2 = newb2

            

        # Calculate Loss and Accuracy for the Training Set
        [z1, z2, h1, yhat] = forward_pass(xTrain, yTrain, W1, W2, b1, b2)
        results['training']['loss'].append(cross_entropy_loss(yTrain, yhat))
        results['training']['acc'].append(accuracy(yTrain, yhat))

        # Calculate Loss and Accuracy for the Validation Set
        [z1, z2, h1, yhat] = forward_pass(xVal, yVal, W1, W2, b1, b2)
        results['validation']['loss'].append(cross_entropy_loss(yVal, yhat))
        results['validation']['acc'].append(accuracy(yVal, yhat))

        # debug_print(epoch, results)
        if epoch > 0 and early_stop(epoch, results):
            in_a_row += 1
        else:
            in_a_row = 0
        
        print("Criterion in a row = {}".format(in_a_row))
        if in_a_row == 3:
            print("Stopped at epoch {}\n".format(epoch + 1))
            last_epoch = epoch + 2
            break

    if in_a_row < 3:
        print("Did Not Stop Early")
        return
    # Calculate Loss and Accuracy for the Training Set
    [z1, z2, h1, yhat] = forward_pass(xTest, yTest, W1, W2, b1, b2)
    test_results['loss'].append(cross_entropy_loss(yTest, yhat))
    test_results['acc'].append(accuracy(yTest, yhat))

    # collect_output(result_list, learning_rates)
    plot_graphs(results, test_results, last_epoch)

def debug_print(epoch, results):

    print("After epoch {}...".format(epoch))
    print("-----Training-----")
    print("\t-----Loss-----")
    print("\t{}".format(results['training']['loss']))
    print("\t-----Accuracy-----")
    print("\t{}".format(results['training']['acc']))
    print("-----Validation-----")
    print("\t-----Loss-----")
    print("\t{}".format(results['validation']['loss']))
    print("\t-----Accuracy-----")
    print("\t{}".format(results['validation']['acc']))
    print()


def early_stop(epoch, results):

    
    if results['training']['loss'][epoch] < results['training']['loss'][epoch - 1] and results['validation']['loss'][epoch] > results['validation']['loss'][epoch - 1]:
        return True

    return False


def collect_output(result_list, learning_rates):

    with open('output.txt', 'a') as f:
        for i in range(0, len(learning_rates)):
            f.write("Learning Rate = {}".format(learning_rates[i]))
            f.write("Training Loss")
            f.write("-------------")
            for train_loss in result_list[i]['training']['loss']:
                f.write(train_loss)
            f.write("Validation Loss")
            f.write("-------------")
            for val_loss in result_list[i]['validation']['loss']:
                f.write(val_loss)
            f.write("")


def plot_graphs(result_list, test_results, end_epoch):


    plt.rcParams["figure.figsize"] = [16,9]
    plt.figure(0)
    plt.plot(list(range(1, end_epoch)), result_list['training']['loss'], 'b', label='Train')
    plt.plot(list(range(1, end_epoch)), result_list['validation']['loss'], 'r', label='Validation')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig('plots/optLoss.png')
    
    plt.figure(1)
    plt.plot(list(range(1, end_epoch)), result_list['training']['acc'], 'b', label='Train')
    plt.plot(list(range(1, end_epoch)), result_list['validation']['acc'], 'r', label='Validation')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('plots/optAcc.png')

    print("At the end of training...")
    print("Training Loss = {}".format(result_list['training']['loss'][-1]))
    print("Validation Loss = {}".format(result_list['validation']['loss'][-1]))
    print("Training Accuracy = {}".format(result_list['training']['acc'][-1]))
    print("Validation Accuracy = {}".format(result_list['validation']['acc'][-1]))
    print("Teting Loss = {}".format(test_results['loss'][-1]))
    print("Testing Accuracy = {}".format(test_results['acc'][-1]))
    


if __name__ == '__main__':

    main()
