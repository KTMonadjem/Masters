/*
 * ANN.h
 *
 *  Created on: Mar 5, 2018
 *      Author: keyan
 */

#ifndef ANN_H_
#define ANN_H_

// includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// definitions
#define MAX_LAYERS 10
#define MAX_NEURONS 10
#define MAX_DATA 1000
#define LEARNING_RATE 0.7
#define LR_STEP 1
#define LR_EPOCHS 100
#define MOMENTUM 0.5

// sigmoid activation functions
//#define tanh(sum) (2.0f/(1.0f + exp(-2.0f * sum)) - 1.0f) // [-1, 1] // math.h -> activation: tanh(double), derivative: pow(1/cosh(double), 2)
#define sigmoid(sum) (1.0f/(1.0f + exp(-1.0f * sum))) // [0, 1]

// ANN type struct
struct ANN{
	int num_layers; // stores number of layers (I + H + ... + O)
	int layers[MAX_LAYERS]; // stores number of neurons per layer (I, H, ..., O)
	float weights[MAX_LAYERS][(int)pow(MAX_NEURONS, 2)]; // stores weight values
	float delta_weights[MAX_LAYERS][(int)pow(MAX_NEURONS, 2)]; // stores previous weight update values for use in momentum calculations
	int max_weights;
	// weights are assigned by future neuron, i.e. the first x weights belong to future neuron 1, the next x to future neuron 2, etc...
	// weights per layer = (neurons in previous layer + 1) * neurons in next layer
	float neurons[MAX_LAYERS][MAX_NEURONS]; // stores neuron output values during calculations (current training values)
	float sums[MAX_LAYERS][MAX_NEURONS]; // stores neuron weighted sum values during calculations (current training values)
	int bias;
	int activation;
};

// train data struct
struct Train_Data{
	int size; // stores the number of samples
	float inputs[MAX_DATA][MAX_NEURONS]; // stores the data inputs
	float outputs[MAX_DATA][MAX_NEURONS]; // stored the data expected outputs
};

// functions
void ann_init(struct ANN * ann, int num_layers, int layers[], int bias, int activation);
void ann_init_custom(struct ANN * ann, int num_layers, int layers[], int max_weights, float weights[][max_weights], int bias, int activation);
void ann_run(float inputs[], float outputs[], struct ANN *ann);
float ann_activation(int activation, float sum);
void ann_train_online(struct ANN *ann, char * filename, int epochs, float error);
void ann_get_deltas_online(struct ANN *ann, float outputs[], float expected_outputs[], int max_weights, float delta_accumulate[][max_weights], float lr);
void ann_train_batch(struct ANN *ann, char * filename, int epochs, float error);
void ann_get_deltas_batch(struct ANN *ann, float outputs[], float expected_outputs[], int max_weights, float delta_accumulate[][max_weights], float lr);
void ann_print(struct ANN *ann, float inputs[], int weights_only);












#endif /* ANN_H_ */
