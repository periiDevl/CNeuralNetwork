#include"Layer.h"
void resetLayer(Layer* layer)
{
    for (size_t i = 0; i < layer->numNeurons; i++)
    {
        resetNeuron(&layer->neurons[i]);
    }
    
}
void setNeuronsSize(Layer* layer, unsigned long long int n)
{
    assert(layer != NULL && "LAYER CANNOT BE NULL!"); //Cannot be NULL at all!
    assert(n > 0 && "YOU CANNOT INPUT 0 NEURONS!");
    if (layer->neurons != NULL) {free(layer->neurons); layer->neurons = NULL;} //Free if not null
    layer->numNeurons = 0;
    layer->neurons = (Neuron*)calloc(n, sizeof(Neuron)); //Allocate size, zero-initialized
    if (layer->neurons == NULL) {
        printf("Error: Allocation FAILED for neurons.\n");
        layer->numNeurons = 0; 
        return;
    }
    layer->numNeurons = n;
    
}



void setUpNeuronsParam(Layer* layer)
{
    for (size_t i = 0; i < layer->numNeurons; i++)
    {
        calNeuronWeightsSize(&layer->neurons[i], layer->next->numNeurons);
    }
}


void freeLayer(Layer* layer){
    if (layer == NULL) {
        return;
    }
    for (size_t i = 0; i < layer->numNeurons; i++)
    {
        freeNeuron(&layer->neurons[i]);
    }
    
    if (layer->neurons != NULL) {
        free(layer->neurons);      
        layer->neurons = NULL;     
    } 
}

void layerForward(Layer* layer)
{
    if (layer == NULL || layer->next == NULL) {return;}
    //Reset
    for (size_t i = 0;i < layer->next->numNeurons; i++)
    {
        layer->next->neurons[i].Z = 0;
    }
    for (size_t i = 0;i < layer->numNeurons; i++)
    {
        double v = layer->neurons[i].val;
        for (size_t j = 0;j < layer->next->numNeurons; j++)
        {
            layer->next->neurons[j].Z += v * layer->neurons[i].weights[j];
        }
    }
    for (size_t i = 0; i < layer->next->numNeurons; i++)
    {
        layer->next->neurons[i].Z += layer->next->neurons[i].bias; // bias added here, so Z is the true pre-activation
        activate(&layer->next->neurons[i]);
    }
}