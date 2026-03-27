#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "Neuron.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include "Backpropagation.h"
#include "MNISTLoader.h"
#include "NetworkIO.h"

#define EPOCHS        50
#define LEARNING_RATE 0.01

double heRandom(int fanIn) {
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double gauss = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265 * u2);
    return gauss * sqrt(2.0 / fanIn);
}

void initWeights(NeuralNetwork* network) {
    for (size_t l = 0; l < network->numLayers - 1; l++) {
        int fanIn = (int)network->layers[l].numNeurons;
        for (size_t n = 0; n < network->layers[l].numNeurons; n++) {
            for (size_t w = 0; w < network->layers[l].neurons[n].weightsSize; w++) {
                network->layers[l].neurons[n].weights[w] = heRandom(fanIn);
            }
        }
    }
}

void shuffle(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

int getPrediction(NeuralNetwork* network, double* confidence) {
    size_t outputLayer = network->numLayers - 1;
    int best = 0;
    double bestVal = network->layers[outputLayer].neurons[0].val;
    for (size_t n = 1; n < network->layers[outputLayer].numNeurons; n++) {
        if (network->layers[outputLayer].neurons[n].val > bestVal) {
            bestVal = network->layers[outputLayer].neurons[n].val;
            best = (int)n;
        }
    }
    if (confidence != NULL) *confidence = bestVal;
    return best;
}


int loadProcessedImage(const char* filename, double* imageBuffer, int size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open %s\n", filename);
        return 0;
    }
    size_t readCount = fread(imageBuffer, sizeof(double), size, file);
    fclose(file);
    if (readCount != size) {
        printf("Error: Read %zu values, expected %d. Invalid file format.\n", readCount, size);
        return 0;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc < 2) {
        printf("Usage:\n");
        printf("  ./mnist train\n");
        printf("  ./mnist predict <input_image.bin>\n");
        return 1;
    }

    // ================== TRAIN MODE ==================
    if (strcmp(argv[1], "train") == 0) {
        double current_lr = LEARNING_RATE;
        // ---- Load MNIST ----
        int trainCount, testCount, labelCount, testLabelCount;
        printf("Loading MNIST...\n");

        double** trainImages = loadMNISTImages("train-images-idx3-ubyte", &trainCount);
        int* trainLabels = loadMNISTLabels("train-labels-idx1-ubyte", &labelCount);
        double** testImages  = loadMNISTImages("t10k-images-idx3-ubyte",  &testCount);
        int* testLabels  = loadMNISTLabels("t10k-labels-idx1-ubyte",  &testLabelCount);

        if (!trainImages || !trainLabels || !testImages || !testLabels) {
            printf("Failed to load MNIST. Check files.\n");
            return 1;
        }

        double** oneHotLabels = toOneHot(trainLabels, trainCount);
        printf("Loaded %d training, %d test samples.\n", trainCount, testCount);

        NeuralNetwork network;
        network.layers = NULL;
        network.numLayers = 0;
        createNeuralNetwork(&network, 4);
        
        // Ensure to link layers before doing parameter sizing
        linkLayers(&network);

        setNeuronsSize(&network.layers[0], 784);
        setNeuronsSize(&network.layers[1], 128);
        setNeuronsSize(&network.layers[2], 64);
        setNeuronsSize(&network.layers[3], 10);

        setLayersParam(&network);
        initWeights(&network);

        int* indices = (int*)malloc(trainCount * sizeof(int));
        for (int i = 0; i < trainCount; i++) indices[i] = i;

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            shuffle(indices, trainCount);
            double totalCost = 0.0;

            for (int i = 0; i < trainCount; i++) {
                int idx = indices[i];
                totalCost += learnFromSample(&network, trainImages[idx],
                                             oneHotLabels[idx], current_lr);

                if ((i + 1) % 5000 == 0) {
                    printf("\rEpoch %d sample %d/%d", epoch + 1, i + 1, trainCount);
                    fflush(stdout);
                }
            }

            //Test accuracy
            int correct = 0;
            for (int i = 0; i < testCount; i++) {
                for (size_t n = 0; n < network.layers[0].numNeurons; n++)
                    network.layers[0].neurons[n].val = testImages[i][n];
                forwardPass(&network);
                if (getPrediction(&network, NULL) == testLabels[i]) correct++;
            }

            printf("\nEpoch %d complete, Avg Cost: %.4f, Test Accuracy: %.2f%%\n",
                   epoch + 1, totalCost / trainCount, (double)correct / testCount * 100.0);
                if ((epoch + 1) % 10 == 0) {
                current_lr *= 0.5;
                printf("Learning rate decayed to: %.5f\n", current_lr);
            }
        }

        // --- NEW: Save the network after training! ---
        saveNetwork(&network, "model.bin");

        // ---- Cleanup ----
        freeMNISTImages(trainImages, trainCount);
        freeMNISTImages(testImages, testCount);
        freeMNISTLabels(trainLabels);
        freeMNISTLabels(testLabels);
        freeOneHot(oneHotLabels, trainCount);
        free(indices);
        freeNetwork(&network);
    } 
    else if (strcmp(argv[1], "predict") == 0) {
        if (argc < 3) {
            printf("Error: Missing image.bin argument.\n");
            return 1;
        }

        NeuralNetwork network;
        network.layers = NULL;
        network.numLayers = 0;

        printf("Loading network from model.bin...\n");
        loadNetwork(&network, "model.bin"); // Loads architecture and weights

        double inputImage[784];
        if (!loadProcessedImage(argv[2], inputImage, 784)) {
            freeNetwork(&network);
            return 1;
        }

        printf("Feeding image into network...\n");
        // Inject image into the input layer
        for (size_t i = 0; i < 784; i++) {
            network.layers[0].neurons[i].val = inputImage[i];
        }

        forwardPass(&network);
        
        double confidence;
        int prediction = getPrediction(&network, &confidence);
        
        printf("Predicted Digit: %d\n", prediction);
        printf("Confidence:      %.2f%%\n", confidence * 100.0);

        freeNetwork(&network);
    } 
    else {
        printf("Unknown command: %s\n", argv[1]);
    }

    return 0;
}
