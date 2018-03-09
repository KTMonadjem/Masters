/*
 * ANN.c
 *
 *  Created on: Mar 5, 2018
 *      Author: keyan
 */

#include "ann.h"

// create NN with random weights
void ann_init(struct ANN * ann, int num_layers, int layers[], int bias, int activation)
{
	int i = 0;
	int j = 0;
	ann->max_weights = 0;
	srand(time(NULL));

	ann->num_layers = num_layers;
	printf("ANN -> Constructing layers.\n");
	for(i = 0; i < num_layers; i++)
	{
		ann->layers[i] = layers[i];
		if(i > 0 && i < num_layers + 1) // assign weights to layers between first and last set of neurons
		{
			int num_weights = layers[i - 1]*layers[i]; // number of weights = no. in previous layer * no. in current layer
			if(bias)
				num_weights += layers[i]; // add the number of weights as neurons in following layer for bias (at last positions)
			for(j = 0; j < num_weights; j++)
			{
				float weight = (float)rand()/(float)(RAND_MAX); // random float between 0 and 1
				ann->weights[i - 1][j] = weight;
			}

			if(ann->layers[i] * ann->layers[i - 1] > ann->max_weights) // reassign max_weights if needed
				ann->max_weights = ann->layers[i] * ann->layers[i - 1];
		}
	}

	printf("ANN -> Applying bias and activation.\n");
	ann->bias = bias;
	ann->activation = activation;
}

// create NN with custom weights
void ann_init_custom(struct ANN * ann, int num_layers, int layers[], int max_weights, float weights[][max_weights], int bias, int activation)
{
	printf("ANN -> Beginning initialization of ANN.\n");
	int i = 0;
	int j = 0;
	srand(time(NULL));

	ann->num_layers = num_layers;
	printf("ANN -> Constructing layers.\n");
	for(i = 0; i < num_layers; i++)
	{
		ann->layers[i] = layers[i];
		if(i > 0 && i < num_layers + 1) // assign weights to layers between first and last set of neurons
		{
			int num_weights = layers[i - 1]*layers[i]; // number of weights = no. in previous layer * no. in current layer
			if(bias)
				num_weights += layers[i]; // add the number of weights as neurons in following layer for bias (at last positions)
			for(j = 0; j < num_weights; j++)
			{
				ann->weights[i - 1][j] = weights[i - 1][j];
			}
		}
	}

	printf("ANN -> Applying bias and activation.\n");
	ann->bias = bias;
	ann->activation = activation;
	ann->max_weights = max_weights;
}

// calculates activation value depending on activation type
float ann_activation(int activation, float sum)
{
	float result;
	switch(activation){
	case 0:
		result = sigmoid(sum);
		break;
	case 1:
		result = tanh(sum);
		break;
	default:
		result = sigmoid(sum);
		break;
	}

	return result;
}

// pass inputs through ANN, i.e. run the ANN
void ann_run(float inputs[], float outputs[], struct ANN *ann)
{
	int i = 0;
	int j = 0;
	int k = 0;

	for(i = 0; i < ann->layers[0]; i++) // run through inputs and add to NN
	{
		//printf("Adding input: %f\n", inputs[i]);
		ann->neurons[0][i] = inputs[i];
		ann->sums[0][i] = 0; // no sums for inputs
	}

	for(i = 1; i < ann->num_layers; i++) // run for every layer except input layer
	{
		//printf("Running through layer: %d\n", i);
		for(j = 0; j < ann->layers[i]; j++) // run through every neuron in current layer
		{
			//printf("Running through current neuron: %d\n", j);
			float weighted_sum = 0;
			for(k = 0; k < ann->layers[i - 1]; k++) // run through every neuron (input) in previous layer
			{
				//printf("Running through previous neuron %d with value: %f\n", k, ann->neurons[i - 1][k]);
				//printf("Multiplying it by weight: %f\n", ann->weights[i - 1][j * ann->layers[i - 1] + k]);
				weighted_sum += ann->neurons[i - 1][k] * ann->weights[i - 1][j * ann->layers[i - 1] + k];
			}

			if(ann->bias) // add bias if necessary (from last position in previous layer)
			{
				//printf("Bias present\n");
				weighted_sum += ann->weights[i - 1][ann->layers[i] * ann->layers[i - 1] + j]; // add bias weight for respective neuron in current layer
			}

			ann->sums[i][j] = weighted_sum; // add weighted sum to sum array for future use
			//printf("Weighted sum is: %f\n", weighted_sum);

			float activation = ann_activation(ann->activation, weighted_sum); // calculate activation result
			ann->neurons[i][j] = activation;
			//printf("Value after activation is: %f\n", activation);
		}
	}

	int num_outputs = ann->layers[ann->num_layers - 1];
	for(i = 0; i < num_outputs; i++) // fill outputs for return
	{
		outputs[i] = ann->neurons[ann->num_layers - 1][i];
	}
}

//BATCH TRAINING -> NOW WORKING
void ann_train_batch(struct ANN *ann, char * filename, int epochs, float error)
{
	int i = 0;
	int j = 0;
	int k = 0;
	int size;
	int num_outputs = ann->layers[ann->num_layers - 1];

	FILE *fp;
	fp = fopen(filename, "r");
	fscanf(fp, "%d", &size);
	if(size > MAX_DATA) // too many samples
		size = MAX_DATA;

	struct Train_Data data; // copying file data to struct to prevent multiple file reads
	for(i = 0; i < size; i++)
	{
		//printf("INPUTS:\n");
		for(j = 0; j < ann->layers[0]; j++) // reading inputs
		{
			fscanf(fp, "%f", &(data.inputs[i][j]));
			//printf("%d[%d]: %f\n", i, j, data.inputs[i][j]);
		}
		//printf("OUTPUTS:\n");
		for(j = 0; j < num_outputs; j++) // reading outputs
		{
			fscanf(fp, "%f", &(data.outputs[i][j]));
			//printf("%d[%d]: %f\n", i, j, data.outputs[i][j]);
		}
	}

	printf("\n======================= TRAINING ======================\n\n");
	printf("ANN -> Training with %d samples, over a maximum of %d epochs and error goal of %f.\n", size, epochs, error);

	// training only variables
	int num_epochs = 0;
	int num_weights = ann->num_layers - 1;
	float delta_accumulate[num_weights][ann->max_weights]; // same size as number of layers of weights
	float mse = 0; // error average of epoch
	float lr = LEARNING_RATE;

	do
	{
		// adapt learning rate (step reduction)
		if(num_epochs%LR_EPOCHS == 0)
		{
			lr *= LR_STEP;
		}
		//printf("ANN -> Current learning rate is %f\n", lr);

		// zero delta_accumulate
		for(j = 0; j < num_weights; j++)
			for(k = 0; k < ann->max_weights; k++)
				delta_accumulate[j][k] = 0;

		mse = 0; // zero mse for addition
		for(i = 0; i < size; i++) // run through full set of data
		{
			float result[num_outputs]; // stores the result of the ann_run
			ann_run(data.inputs[i], result, ann); // run ANN with selected inputs

			for(j = 0; j < num_outputs; j++) // run through outputs and get error
			{
				float output_error = data.outputs[i][j] - result[j]; // calculate output error
				mse += pow(output_error, 2); // add squared error to mse
			}

			ann_get_deltas_batch(ann, result, data.outputs[i], ann->max_weights, delta_accumulate, lr);
		}

		// calculate error
		mse /= (size * num_outputs); // divide error sum by total number of outputs
		num_epochs++;


		// average weight deltas and correct weights
		for(i = 0; i < num_weights; i++) // run through each layer
		{
			//printf("ANN -> Updating %d weights\n", ann->layers[i] * ann->layers[i + 1]);
			for(j = 0; j < ann->layers[i] * ann->layers[i + 1]; j++) // run through each delta weight sum
			{
				float delta_weight = delta_accumulate[i][j]/(float)size;  // average delta accumulate by dividing by number of training samples
				//float delta_weight = delta_accumulate[i][j]; // DO NOT AVERAGE THE WEIGHTS
				//printf("ANN -> delta_weight is: %f/%f = %f\n", delta_accumulate[i][j], (float)size, delta_weight);
				ann->weights[i][j] += delta_weight; // add to the corresponding weight
			}

			// add to bias neuron if necessary
			if(ann->bias)
			{
				for(j = 0; j < ann->layers[i + 1]; j++) // run through each bias weight at end of the weights
				{
					// update bias' in the positions of the final weights
					ann->weights[i][ann->layers[i] * ann->layers[i + 1] + j] = delta_accumulate[i][ann->layers[i] * ann->layers[i + 1] + j]/(float)size;
				}
			}
		}

		printf("EPOCH: %d		MSE: %f		LEARNING RATE: %f\n", num_epochs, mse, lr);
	}
	while(epochs > num_epochs && mse > error);

	printf("\n");
	printf("\n============= FINISHED TRAINING ==============\n\n");

	fclose(fp);
}

// helper function to get the delta values of a single pass
void ann_get_deltas_batch(struct ANN *ann, float outputs[], float expected_outputs[], int max_weights, float delta_accumulate[][max_weights], float lr)
{
	int i = 0;
	int j = 0;
	int num_weights = ann->num_layers - 1;
	int layer = num_weights; // start at output layer
	float delta_sums[num_weights][max_weights]; // delta_sums
	float learning_rate = lr;

	for(i = 0; i < ann->layers[num_weights]; i++) // transform output layer into initial delta_sum
	{
		delta_sums[num_weights - 1][i] = expected_outputs[i] - outputs[i]; // calculate output error
		//printf("Output: %f\nDesired output: %f\n", outputs[i], expected_outputs[i]);
	}

	for(; layer > 0; layer--) // iterate through each layer, calculating the delta_sum and adding them to delta_sums
	{
		//printf("ANN -> In layer %d\n", layer);
		for(i = 0; i < ann->layers[layer]; i++) // run through each neuron in the current layer
		{
			//printf("ANN -> In current neuron %d\n", i);
			for(j = 0; j < ann->layers[layer - 1]; j++) // run through each neuron in the previous layer
			{
				//printf("ANN -> In previous neuron %d\n", j);
				if(layer > 1)
				{
					if(i == 0) // first neuron, so zero the delta_sums in previous layer
						delta_sums[layer - 2][j] = 0;
					// delta_sum i = wij * delta_j + wik * delta_k + ...
					//printf("ANN -> Adding %f * %f = %f to the current delta sum of %f\n", delta_sums[layer - 1][i],
					//					ann->weights[layer - 1][i * ann->layers[layer - 1] + j],
					//					delta_sums[layer - 1][i] * ann->weights[layer - 1][i * ann->layers[layer - 1] + j], delta_sums[layer - 2][j]);
					delta_sums[layer - 2][j] += delta_sums[layer - 1][i] * ann->weights[layer - 1][i * ann->layers[layer - 1] + j]; // add to delta_sums for current layer

					//printf("ANN -> Delta sum for previous neuron is currently %f\n", delta_sums[layer - 2][j]);
				}

				// at the same time, calculate weight updates for this current layer using the previous layer's delta values
				// weight update w'ij = learning_rate * delta_j * dy_i/d_sum * y_i
				// for now, default learning rate is 0.7

				// calculate gradient of error
				float error_gradient;
				switch(ann->activation){
				case 0: // differentiate sigmoid(x) = f(x): f'(x) = f(x)[1 - f(x)]
					error_gradient = ann->neurons[layer][i] * (1 - ann->neurons[layer][i]); // delta_sum = sigmoid(sum)(1 - sigmoid(sum))*output_error
					//rintf("ANN -> Error gradient of current neuron is %f * (1 - %f) = %f\n", ann->neurons[layer][i], ann->neurons[layer][i], error_gradient);
					break;
				case 1: // differentiate tanh(x) = f(x): f'(x) = sech(x)^2 = 1/cosh(x)^2
					error_gradient = pow(1.0/cosh(ann->sums[layer][i]), 2); // delta_sum = (1/cosh(sum))^2 * output_error (ann->sums starts at 1, not 0)
					//printf("ANN -> Error gradient of current neuron is (1/cosh(%f))^2 = (1/%f)^2 = %f\n", ann->sums[layer][i], cosh(ann->sums[layer][i]), error_gradient);
					break;
				default: // differentiate sigmoid(x) = f(x): f'(x) = f(x)[1 - f(x)]
					error_gradient = ann->neurons[layer][i] * (1 - ann->neurons[layer][i]); // delta_sum = sigmoid(sum)(1 - sigmoid(sum))*output_error
					break;
				}

				// calculate weight update
				float weight_update = learning_rate * delta_sums[layer - 1][i] * error_gradient * ann->neurons[layer - 1][j];
				//printf("ANN -> Adding to accumulate delta weight %d by %f * %f * %f * %f = %f\n", (i * ann->layers[layer - 1] + j), LEARNING_RATE, delta_sums[layer - 1][i],
				//		error_gradient, ann->neurons[layer - 1][j], weight_update);

				delta_accumulate[layer - 1][i * ann->layers[layer - 1] + j] += weight_update;
			}

			// calculate bias delta weights
			if(ann->bias)
			{
				// update bias weight and add to accumulate for bias connection to this neuron
				float weight_update = learning_rate * delta_sums[layer - 1][i]; // no dy_i/d_sum since output is always 1
				delta_accumulate[layer - 1][ann->layers[layer] * ann->layers[layer - 1] + i] += weight_update; // add to last position in the weights layer
			}
		}
	}
}

// display the ANN
void ann_print(struct ANN *ann, float inputs[], int weights_only)
{
	int i = 0;
	int j = 0;
	int k = 0;

	printf("\n+++++++++++++++ PRINTING NEURAL NETWORK STRUCTURE +++++++++++++++\n");
	printf("NUMBER OF LAYERS: %d\n", ann->num_layers);
	printf("NUMBER OF INPUTS: %d\n", ann->layers[0]);
	printf("NUMBER OF OUTPUTS: %d\n", ann->layers[ann->num_layers - 1]);
	printf("NEURONS PER LAYER: %d", ann->layers[0]);
	for(i = 1; i < ann->num_layers; i++)
	{
		printf(" -> %d", ann->layers[i]);
	}
	printf("\n");
	if(ann->bias)
		printf("THIS NEURAL NETWORK HAS BIAS NEURONS\n");
	else
		printf("THIS NEURAL NETWORK DOES NOT HAVE BIAS NEURONS\n");
	printf("ACTIVATION FUNCTION: ");
	switch(ann->activation){
	case 0:
		printf("SIGMOID\n");
		break;
	case 1:
		printf("TANH (SYMMETRIC SIGMOID)\n");
		break;
	default:
		printf("SIGMOID\n");
		break;
	}


	if(!weights_only) // runs the neural network to get layer values if necessary
	{
		float outputs[ann->layers[ann->num_layers - 1]];
		ann_run(inputs, outputs, ann);
	}

	for(i = 0; i < ann->num_layers - 1; i++) // run through all layers except last layer
	{
		printf("\n<============ Listing LAYER %d ============>\n\n", i);
		for(j = 0; j < ann->layers[i]; j++) // run through all neurons in current layer
		{
			if(!weights_only) // only prints layer values if necessary
				printf("LAYER %d NEURON %d has a WEIGHTED INPUT SUM of %f and an ACTIVATION OUTPUT of %f\n", i, j, ann->sums[i][j], ann->neurons[i][j]);
			printf("LAYER %d NEURON %d has %d connections to LAYER %d:\n", i, j, ann->layers[i + 1], (i + 1));
			for(k = 0; k < ann->layers[i + 1]; k++) // through all neurons in following layer
			{
				printf("-> Connection to NEURON %d in LAYER %d has a WEIGHT of %f\n", k, (i + 1), ann->weights[i][k * ann->layers[i] + j]);
			}
		}
		if(ann->bias)
		{
			printf("LAYER %d has a BIAS NEURON with %d connections\n", i, ann->layers[i + 1]);
			for(j = 0; j < ann->layers[i + 1]; j++) // run through bias neurons at the end
			{
				printf("-> Connection to NEURON %d in LAYER %d has a WEIGHT of %f\n", j, (i + 1), ann->weights[i][ann->layers[i] * ann->layers[i + 1] + j]);
			}
		}
	}
	// display output layer
	if(!weights_only) // only finds output layer if necessary
	{
		printf("\n<============ Listing OUTPUT LAYER ============>\n\n");
		int output_layer = ann->num_layers - 1;
		for(i = 0; i < ann->layers[output_layer]; i++)
		{
			printf("LAYER %d NEURON %d has a WEIGHTED INPUT SUM of %f and an ACTIVATION OUTPUT of %f\n",
					output_layer, i, ann->sums[output_layer][i], ann->neurons[output_layer][i]);
		}
	}
}




// online ANN training
void ann_train_online(struct ANN *ann, char * filename, int epochs, float error)
{
	int i = 0;
	int j = 0;
	int k = 0;
	int size;
	int num_outputs = ann->layers[ann->num_layers - 1];

	FILE *fp;
	fp = fopen(filename, "r");
	fscanf(fp, "%d", &size);
	if(size > MAX_DATA) // too many samples
		size = MAX_DATA;

	struct Train_Data data; // copying file data to struct to prevent multiple file reads
	for(i = 0; i < size; i++)
	{
		//printf("INPUTS:\n");
		for(j = 0; j < ann->layers[0]; j++) // reading inputs
		{
			fscanf(fp, "%f", &(data.inputs[i][j]));
			//printf("%d[%d]: %f\n", i, j, data.inputs[i][j]);
		}
		//printf("OUTPUTS:\n");
		for(j = 0; j < num_outputs; j++) // reading outputs
		{
			fscanf(fp, "%f", &(data.outputs[i][j]));
			//printf("%d[%d]: %f\n", i, j, data.outputs[i][j]);
		}
	}

	printf("\n======================= TRAINING ======================\n\n");
	printf("ANN -> Training with %d samples, over a maximum of %d epochs and error goal of %f.\n", size, epochs, error);

	// training only variables
	int num_epochs = 0;
	int num_weights = ann->num_layers - 1;
	float delta_accumulate[num_weights][ann->max_weights]; // same size as number of layers of weights
	float mse = 0; // error average of epoch
	float lr = LEARNING_RATE;

	do
	{
		// adapt learning rate (step reduction)
		if(num_epochs%LR_EPOCHS == 0)
		{
			lr *= LR_STEP;
		}
		mse = 0; // zero mse for addition
		for(i = 0; i < size; i++) // run through full set of data
		{
			float result[num_outputs]; // stores the result of the ann_run
			ann_run(data.inputs[i], result, ann); // run ANN with selected inputs

			for(j = 0; j < num_outputs; j++) // run through outputs and get error
			{
				float output_error = data.outputs[i][j] - result[j]; // calculate output error
				mse += pow(output_error, 2); // add squared error to mse
			}

			ann_get_deltas_online(ann, result, data.outputs[i], ann->max_weights, delta_accumulate, lr);

			// update weights
			// average weight deltas and correct weights
			for(j = 0; j < num_weights; j++) // run through each layer
			{
				//printf("ANN -> Updating %d weights\n", ann->layers[i] * ann->layers[i + 1]);
				for(k = 0; k < ann->layers[j] * ann->layers[j + 1]; k++) // run through each delta weight
				{
					ann->weights[j][k] += delta_accumulate[j][k]; // update weight
				}
				// add to bias neuron if necessary
				if(ann->bias)
				{
					for(k = 0; k < ann->layers[j + 1]; k++) // run through each bias weight at end of the weights
					{
						// update bias' in the positions of the final weights
						ann->weights[j][ann->layers[j] * ann->layers[j + 1] + k] = delta_accumulate[j][ann->layers[j] * ann->layers[j + 1] + k]/(float)size;
					}
				}
			}
		}

		// calculate error
		mse /= (size * num_outputs); // divide error sum by total number of outputs
		num_epochs++;

		printf("EPOCH: %d		MSE: %f		LEARNING RATE: %f\n", num_epochs, mse, lr);
	}
	while(epochs > num_epochs && mse > error);

	printf("\n");
	printf("\n============= FINISHED TRAINING ==============\n\n");

	fclose(fp);
}

// helper function to get the delta values of a single pass
void ann_get_deltas_online(struct ANN *ann, float outputs[], float expected_outputs[], int max_weights, float delta_accumulate[][max_weights], float lr)
{
	int i = 0;
	int j = 0;
	int num_weights = ann->num_layers - 1;
	int layer = num_weights; // start at output layer
	float delta_sums[num_weights][max_weights]; // delta_sums
	float learning_rate = lr;

	for(i = 0; i < ann->layers[num_weights]; i++) // transform output layer into initial delta_sum
	{
		delta_sums[num_weights - 1][i] = expected_outputs[i] - outputs[i]; // calculate output error
		//printf("Output: %f\nDesired output: %f\n", outputs[i], expected_outputs[i]);
	}

	for(; layer > 0; layer--) // iterate through each layer, calculating the delta_sum and adding them to delta_sums
	{
		//printf("ANN -> In layer %d\n", layer);
		for(i = 0; i < ann->layers[layer]; i++) // run through each neuron in the current layer
		{
			//printf("ANN -> In current neuron %d\n", i);
			for(j = 0; j < ann->layers[layer - 1]; j++) // run through each neuron in the previous layer
			{
				//printf("ANN -> In previous neuron %d\n", j);

				if(layer > 1)
				{
					if(i == 0) // first neuron, so zero the delta_sums in previous layer
						delta_sums[layer - 2][j] = 0;
					// delta_sum i = wij * delta_j + wik * delta_k + ...
					//printf("ANN -> Adding %f * %f = %f to the current delta sum of %f\n", delta_sums[layer - 1][i],
					//					ann->weights[layer - 1][i * ann->layers[layer - 1] + j],
					//					delta_sums[layer - 1][i] * ann->weights[layer - 1][i * ann->layers[layer - 1] + j], delta_sums[layer - 2][j]);
					delta_sums[layer - 2][j] += delta_sums[layer - 1][i] * ann->weights[layer - 1][i * ann->layers[layer - 1] + j]; // add to delta_sums for current layer

					//printf("ANN -> Delta sum for previous neuron is currently %f\n", delta_sums[layer - 2][j]);
				}

				// at the same time, calculate weight updates for this current layer using the previous layer's delta values
				// weight update w'ij = learning_rate * delta_j * dy_i/d_sum * y_i
				// for now, default learning rate is 0.7

				// calculate gradient of error
				float error_gradient;
				switch(ann->activation){
				case 0: // differentiate sigmoid(x) = f(x): f'(x) = f(x)[1 - f(x)]
					error_gradient = ann->neurons[layer][i] * (1 - ann->neurons[layer][i]); // delta_sum = sigmoid(sum)(1 - sigmoid(sum))*output_error
					//rintf("ANN -> Error gradient of current neuron is %f * (1 - %f) = %f\n", ann->neurons[layer][i], ann->neurons[layer][i], error_gradient);
					break;
				case 1: // differentiate tanh(x) = f(x): f'(x) = sech(x)^2 = 1/cosh(x)^2
					error_gradient = pow(1.0/cosh(ann->sums[layer][i]), 2); // delta_sum = (1/cosh(sum))^2 * output_error (ann->sums starts at 1, not 0)
					//printf("ANN -> Error gradient of current neuron is (1/cosh(%f))^2 = (1/%f)^2 = %f\n", ann->sums[layer][i], cosh(ann->sums[layer][i]), error_gradient);
					break;
				default: // differentiate sigmoid(x) = f(x): f'(x) = f(x)[1 - f(x)]
					error_gradient = ann->neurons[layer][i] * (1 - ann->neurons[layer][i]); // delta_sum = sigmoid(sum)(1 - sigmoid(sum))*output_error
					break;
				}

				// calculate weight update
				float weight_update = learning_rate * delta_sums[layer - 1][i] * error_gradient * ann->neurons[layer - 1][j];
				//printf("ANN -> Making accumulate delta weight %d by %f * %f * %f * %f = %f\n", (i * ann->layers[layer - 1] + j), LEARNING_RATE, delta_sums[layer - 1][i],
				//		error_gradient, ann->neurons[layer - 1][j], weight_update);

				delta_accumulate[layer - 1][i * ann->layers[layer - 1] + j] = weight_update;
			}

			// calculate bias delta weights
			if(ann->bias)
			{
				// update bias weight and add to accumulate for bias connection to this neuron
				float weight_update = learning_rate * delta_sums[layer - 1][i]; // no dy_i/d_sum since output is always 1
				delta_accumulate[layer - 1][ann->layers[layer] * ann->layers[layer - 1] + i] += weight_update; // add to last position in the weights layer
			}
		}
	}
}











