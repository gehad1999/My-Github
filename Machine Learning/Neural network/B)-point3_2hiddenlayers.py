import numpy as np
import random
import math
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, numInput, numHidden, numHidden2, numOutput, seed):
        self.ni = numInput
        self.nh = numHidden
        self.nh2 = numHidden2  #####################
        self.no = numOutput

        self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)

        self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
        self.hNodes2 = np.zeros(shape=[self.nh2], dtype=np.float32)  ##############33

        self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)

        self.ihWeights = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        self.hhWeights = np.zeros(shape=[self.nh, self.nh2], dtype=np.float32)  ################
        self.hoWeights = np.zeros(shape=[self.nh2, self.no], dtype=np.float32)  #############

        self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
        self.hBiases2 = np.zeros(shape=[self.nh2], dtype=np.float32)  #################33

        self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)
        self.hhGrads = np.zeros(shape=[self.nh, self.nh2], dtype=np.float32)  #########################
        self.hoGrads = np.zeros(shape=[self.nh2, self.no],
                                dtype=np.float32)  # hidden-to-output weights gradients############
        self.obGrads = np.zeros(shape=[self.no], dtype=np.float32)  # output node biases gradients
        self.ihGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)  # input-to-hidden weights gradients
        self.hbGrads = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden biases gradients
        self.hbGrads2 = np.zeros(shape=[self.nh2], dtype=np.float32)  # hidden biases gradients#################

        self.oSignals = np.zeros(shape=[self.no], dtype=np.float32)  # output signals: gradients w/o assoc. input terms
        self.hSignals = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden signals: gradients w/o assoc. input terms
        self.hSignals2 = np.zeros(shape=[self.nh2], dtype=np.float32)  ####################

        self.ih_prev_weights_delta = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)  # momentum
        self.h_prev_biases_delta = np.zeros(shape=[self.nh], dtype=np.float32)
        self.h_prev_biases_delta2 = np.zeros(shape=[self.nh2], dtype=np.float32)  #######################
        self.hh_prev_weights_delta = np.zeros(shape=[self.nh, self.nh2], dtype=np.float32)  ############
        self.ho_prev_weights_delta = np.zeros(shape=[self.nh2, self.no], dtype=np.float32)  ##############
        self.o_prev_biases_delta = np.zeros(shape=[self.no], dtype=np.float32)
        self.x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        self.t_values = np.zeros(shape=[self.no], dtype=np.float32)
        self.rnd = random.Random(seed)  # allows multiple instances
        self.initializeWeights()

    def Update_Weights(self):
        tw = (self.ni * self.nh) + (self.nh * self.nh2) + (self.nh2 * self.no) + self.nh2 + self.nh + self.no
        result = np.zeros(shape=[tw], dtype=np.float32)

        idx = 0  # points into result

        for i in range(self.ni):
            for j in range(self.nh):
                result[idx] = self.ihWeights[i, j]
                idx += 1

        for i in range(self.nh):
            for j in range(self.nh2):
                result[idx] = self.hhWeights[i, j]
                idx += 1

        for j in range(self.nh):
            result[idx] = self.hBiases[j]
            idx += 1

        for j in range(self.nh2):
            result[idx] = self.hBiases2[j]
            idx += 1

        for i in range(self.nh2):
            for k in range(self.no):
                result[idx] = self.hoWeights[i, k]
                idx += 1

        for k in range(self.no):
            result[idx] = self.oBiases[k]
            idx += 1

        return result

    def initializeWeights(self):
        numWts = (self.ni * self.nh) + (self.nh * self.nh2) + (self.nh2 * self.no) + self.nh2 + self.nh + self.no
        wts = np.zeros(shape=[numWts], dtype=np.float32)
        lo = -0.01;
        hi = 0.01
        for idx in range(len(wts)):
            wts[idx] = (hi - lo) * self.rnd.random() + lo

        if len(wts) != numWts:
            print("Warning: len(weights) error in Weights()")

        idx = 0
        for i in range(self.ni):
            for j in range(self.nh):
                self.ihWeights[i, j] = wts[idx]
                idx += 1

        for j in range(self.nh):
            self.hBiases[j] = wts[idx]
            idx += 1

        for i in range(self.nh):
            for j in range(self.nh2):
                self.hhWeights[i, j] = wts[idx]
                idx += 1

        for j in range(self.nh2):
            self.hBiases2[j] = wts[idx]
            idx += 1

        for i in range(self.nh2):
            for j in range(self.no):
                self.hoWeights[i, j] = wts[idx]
                idx += 1

        for k in range(self.no):
            self.oBiases[k] = wts[idx]
            idx += 1

    def forward_propagation(self, xValues):
        hSums = np.zeros(shape=[self.nh], dtype=np.float32)
        hSums2 = np.zeros(shape=[self.nh2], dtype=np.float32)
        oSums = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(self.ni):
            self.iNodes[i] = xValues[i]

        for j in range(self.nh):
            for i in range(self.ni):
                hSums[j] += self.iNodes[i] * self.ihWeights[i, j]

        for j in range(self.nh):
            hSums[j] += self.hBiases[j]

        for j in range(self.nh):
            self.hNodes[j] = self.activation_func_hypertan(hSums[j])

        for j in range(self.nh2):
            for i in range(self.nh):
                hSums2[j] += self.hNodes[i] * self.hhWeights[i, j]

        for j in range(self.nh2):
            hSums2[j] += self.hBiases2[j]

        for j in range(self.nh2):
            self.hNodes2[j] = self.activation_func_hypertan(hSums2[j])

        for k in range(self.no):
            for j in range(self.nh2):
                oSums[k] += self.hNodes2[j] * self.hoWeights[j, k]

        for k in range(self.no):
            oSums[k] += self.oBiases[k]

        softOut = self.activation_func_softmax(oSums)

        for k in range(self.no):
            self.oNodes[k] = softOut[k]
        result = np.zeros(shape=self.no, dtype=np.float32)
        for k in range(self.no):
            result[k] = self.oNodes[k]
        return result

    def back_propagation(self, learnRate, momentum):
        for k in range(self.no):
            derivative = (1 - self.oNodes[k]) * self.oNodes[k]  # softmax
            self.oSignals[k] = derivative * (self.t_values[k] - self.oNodes[k])  # target - output => add delta

        # 2. compute hidden-to-output weight gradients using output signals
        for j in range(self.nh2):
            for k in range(self.no):
                self.hoGrads[j, k] = self.oSignals[k] * self.hNodes2[j]

        # 3. compute output node bias gradients using output signals
        for k in range(self.no):
            self.obGrads[k] = self.oSignals[k] * 1.0  # 1.0 dummy input can be dropped

        # 4. compute hidden node signals
        for j in range(self.nh2):
            sum = 0.0
            for k in range(self.no):
                sum += self.oSignals[k] * self.hoWeights[j, k]
            derivative = (1 - self.hNodes2[j]) * (1 + self.hNodes2[j])  # tanh activation
            self.hSignals2[j] = derivative * sum

            # 2. compute hidden-to-output weight gradients using output signals
        for j in range(self.nh):
            for k in range(self.nh2):
                self.hhGrads[j, k] = self.hSignals2[k] * self.hNodes[j]

        # 3. compute output node bias gradients using output signals
        for k in range(self.nh2):
            self.hbGrads2[k] = self.hSignals2[k] * 1.0  # 1.0 dummy input can be dropped

        # 4. compute hidden node signals
        for j in range(self.nh):
            sum = 0.0
            for k in range(self.nh2):
                sum += self.hSignals2[k] * self.hhWeights[j, k]
            derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])  # tanh activation
            self.hSignals[j] = derivative * sum

        # 5 compute input-to-hidden weight gradients using hidden signals
        for i in range(self.ni):
            for j in range(self.nh):
                self.ihGrads[i, j] = self.hSignals[j] * self.iNodes[i]

        # 6. compute hidden node bias gradients using hidden signals
        for j in range(self.nh):
            self.hbGrads[j] = self.hSignals[j] * 1.0  # 1.0 dummy input can be dropped

        # update weights and biases using the gradients

        # 1. update input-to-hidden weights
        for i in range(self.ni):
            for j in range(self.nh):
                delta = learnRate * self.ihGrads[i, j]
                self.ihWeights[i, j] += delta
                self.ihWeights[i, j] += momentum * self.ih_prev_weights_delta[i, j]
                self.ih_prev_weights_delta[i, j] = delta  # save the delta for next iteration

        # 2. update hidden node biases
        for j in range(self.nh):
            delta = learnRate * self.hbGrads[j] * 1.0  # can drop the dummy 1.0 input
            self.hBiases[j] += delta
            self.hBiases[j] += momentum * self.h_prev_biases_delta[j]
            self.h_prev_biases_delta[j] = delta  # save the delta

        for j in range(self.nh2):
            delta = learnRate * self.hbGrads2[j] * 1.0  # can drop the dummy 1.0 input
            self.hBiases2[j] += delta
            self.hBiases2[j] += momentum * self.h_prev_biases_delta2[j]
            self.h_prev_biases_delta2[j] = delta  # save the delta

        # 3. update hidden-to-output weights
        for j in range(self.nh):
            for k in range(self.nh2):
                delta = learnRate * self.hhGrads[j, k]
                self.hhWeights[j, k] += delta
                self.hhWeights[j, k] += momentum * self.hh_prev_weights_delta[j, k]
                self.hh_prev_weights_delta[j, k] = delta  # save the delta

        for j in range(self.nh2):
            for k in range(self.no):
                delta = learnRate * self.hoGrads[j, k]
                self.hoWeights[j, k] += delta
                self.hoWeights[j, k] += momentum * self.ho_prev_weights_delta[j, k]
                self.ho_prev_weights_delta[j, k] = delta  # save the delta

        # 4. update output node biases
        for k in range(self.no):
            delta = learnRate * self.obGrads[k]
            self.oBiases[k] += delta
            self.oBiases[k] += momentum * self.o_prev_biases_delta[k]
            self.o_prev_biases_delta[k] = delta  # save the delta

    def train(self, trainData, maxEpochs, learnRate, momentum):
        epoch = 0
        errors = []
        accs = []
        # x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        # t_values = np.zeros(shape=[self.no], dtype=np.float32)
        numTrainItems = len(trainData)
        indices = np.arange(numTrainItems)  # [0, 1, 2, . . n-1]  # rnd.shuffle(v)

        while epoch < maxEpochs:
            self.rnd.shuffle(indices)  # scramble order of training items
            for ii in range(numTrainItems):
                idx = indices[ii]

                for j in range(self.ni):
                    self.x_values[j] = trainData[idx, j]  # get the input values
                for j in range(self.no):
                    self.t_values[j] = trainData[idx, j + self.ni]  # get the target values
                self.forward_propagation(self.x_values)  # results stored internally
                self.back_propagation(learnRate, momentum)

            epoch += 1

            if epoch % 100 == 0:
                mse = self.meanSquaredError(trainData)
                accu = self.accuracy(trainData)
                print("epoch = " + str(epoch) + " ms error = %0.4f " % mse)
                print("epoch = " + str(epoch) + " ms accu = %0.4f " % accu)
                errors.append(mse)
                accs.append(1 - mse)
        # end while

        result = self.Update_Weights()
        return result, accs, errors

    # end train

    def accuracy(self, tdata):  # train or test data matrix
        num_correct = 0;
        num_wrong = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(len(tdata)):  # walk thru each data item
            for j in range(self.ni):  # peel off input values from curr data row
                x_values[j] = tdata[i, j]
            for j in range(self.no):  # peel off tareget values from curr data row
                t_values[j] = tdata[i, j + self.ni]

            y_values = self.forward_propagation(x_values)  # computed output values)
            max_index = np.argmax(y_values)  # index of largest output value

            if abs(t_values[max_index] - 1.0) < 1.0e-5:
                num_correct += 1
            else:
                num_wrong += 1

        return (num_correct * 1.0) / (num_correct + num_wrong)

    def meanSquaredError(self, tdata):  # on train or test data matrix
        sumSquaredError = 0.0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(len(tdata)):  # walk through each data item
            for j in range(self.ni):  # peel off input values from curr data row
                x_values[j] = tdata[i, j]

            for j in range(self.no):  # peel off tareget values from curr data row
                t_values[j] = tdata[i, j + self.ni]

            y_values = self.forward_propagation(x_values)  # computed output values

            for j in range(self.no):
                err = t_values[j] - y_values[j]
                sumSquaredError += err * err
        return sumSquaredError / len(tdata)

    @staticmethod
    def activation_func_hypertan(x):
        if x < -20.0:
            return -1.0
        elif x > 20.0:
            return 1.0
        else:
            return math.tanh(x)

    @staticmethod
    def activation_func_softmax(oSums):
        result = np.zeros(shape=[len(oSums)], dtype=np.float32)
        m = max(oSums)
        divisor = 0.0
        for k in range(len(oSums)):
            divisor += math.exp(oSums[k] - m)
        for k in range(len(result)):
            result[k] = math.exp(oSums[k] - m) / divisor
        return result

# end class NeuralNetwork
def main():
    numInput = 12
    numHidden = 6
    numHidden2 = 4
    numOutput = 11
    seed = 3
    print("Creating a %d-%d-%d-%d neural network " % (numInput, numHidden, numHidden2, numOutput))
    nn = NeuralNetwork(numInput, numHidden, numHidden2, numOutput, seed)

    print("\nLoading vowel training and test data ")
    trainDataPath = "vowel_train.txt"
    trainDataMatrix = np.loadtxt(trainDataPath, dtype=np.float32, delimiter=",")

    testDataPath = "vowel_test.txt"
    testDataMatrix = np.loadtxt(testDataPath, dtype=np.float32, delimiter=",")

    maxEpochs = 500
    learnRate = 0.01
    momentum = 0.75

    print("\nSetting maxEpochs = " + str(maxEpochs))
    print("Setting learning rate = %0.3f " % learnRate)
    print("Setting momentum = %0.3f " % momentum)

    print("\nStarting training without momentum")
    result, accu, errors = nn.train(trainDataMatrix, maxEpochs, learnRate, 0.0)
    print("Training complete")

    epochs = [0, 1, 2, 3, 4]
    plt.plot(epochs, errors)
    plt.xlabel("epochs ")
    plt.ylabel('error')
    plt.show()

    plt.plot(epochs, accu)
    plt.xlabel("epochs ")
    plt.ylabel('accu')
    plt.show()
    accTrain = nn.accuracy(trainDataMatrix)
    accTest = nn.accuracy(testDataMatrix)

    print("\nAccuracy on 742-item train data = %0.4f " % accTrain)
    print("Accuracy on 248-item test data   = %0.4f " % accTest)

    print("\nEnd demo \n")


if __name__ == "__main__":
    main()

# end script


