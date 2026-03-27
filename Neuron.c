#include"Neuron.h"
#include <stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
void resetNeuron(Neuron* neuron)
{
    neuron->Z = 0.0; 
    for (size_t i = 0; i < neuron->weightsSize; i++)
    {
        neuron->weightsGradients[i] = 0.0;
    }
}
void calNeuronWeightsSize(Neuron* neuron, unsigned long long int n)
{
    assert(neuron != NULL && "NEURON CANNOT BE NULL!"); //Cannot be NULL at all!
    if (neuron->weights != NULL) {free(neuron->weights); neuron->weights = NULL;} //Free if not null
    if (neuron->weightsGradients != NULL) {free(neuron->weightsGradients); neuron->weightsGradients = NULL;}
    neuron->weightsSize = 0;
    neuron->weights = (double*)malloc(n*sizeof(double)); //Allocate size
    neuron->weightsGradients = (double*)malloc(n*sizeof(double)); //Allocate size

    if (neuron->weights == NULL) {
        printf("Error: Allocation FAILED for neuron weights.\n");
        neuron->weightsSize = 0; 
        return;
    }
    neuron->weightsSize = n;
    for (size_t i = 0; i < n; i++) {
        neuron->weights[i] = 0.0;
        neuron->weightsGradients[i] = 0.0;
    }
    neuron->bias = 0.0;
    neuron->biasGradient = 0.0;
    neuron->Z = 0.0;
    neuron->val = 0.0;
}
double ELUactivation(double x) {
    if (x > 0) {
        return x;
    } else {
        return 1.0 * (exp(x) - 1.0);
    }
}
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
void activate(Neuron* neuron)
{
    neuron->val = sigmoid(neuron->Z);
}
void freeNeuron(Neuron* neuron) {
    if (neuron == NULL) {
        printf("Warning: Attempted to free memory for a NULL Neuron pointer. No action taken.\n");
        return;
    }
    if (neuron->weights != NULL) {
        free(neuron->weights);
        neuron->weights = NULL;
    }
    if (neuron->weightsGradients != NULL) {
        free(neuron->weightsGradients);
        neuron->weightsGradients = NULL;
    }
}