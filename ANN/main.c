/*
 * main.c
 *
 *  Created on: Mar 5, 2018
 *      Author: keyan
 */

#include "ann.c"

int main(int argc, char **argv)
{
	// Test example with bp
//	int num_layers = 3;
//	int layers[3] = {2, 3, 1};
//	int max_weights = 6;
//	float weights[2][6] = {{0.8, 0.2, 0.4, 0.9, 0.3, 0.5}, {0.3, 0.5, 0.9, 0, 0, 0}};
//	int bias = 0;
//	int activation = 0;
//	struct ANN new_ann;
//	struct ANN *my_ann = &new_ann;
//	ann_init_custom(my_ann, num_layers, layers, max_weights, weights, bias, activation);
//
//	float inputs[2] = {1, 1};
//	float outputs[1];
//	ann_run(inputs, outputs, my_ann);
//
//	ann_print(my_ann, inputs, 0);
//	//printf("The ANN took in %f and %f and returned %f.\n", inputs[0], inputs[1], outputs[0]);
//
//	ann_train(my_ann, "test.data", 100, 0.01);
//
//	ann_print(my_ann, inputs, 0);

	int num_layers = 3;
	int layers[3] = {2, 4, 1};
	int bias = 1;
	int activation = 1;
	struct ANN new_ann;
	struct ANN *my_ann = &new_ann;
	ann_init(my_ann, num_layers, layers, bias, activation);

	float inputs[2] = {1, 1};
	float outputs[1];
	//ann_run(inputs, outputs, my_ann);

	ann_print(my_ann, inputs, 1);
	//printf("The ANN took in %f and %f and returned %f.\n", inputs[0], inputs[1], outputs[0]);

	ann_train_batch(my_ann, "train.data", 1000, 0.0001);

	//ann_print(my_ann, inputs, 1);

	int i = 0;
	int score = 0;
	float err_av = 0;
	for(; i < 100; i++)
	{
		inputs[0] = (float)rand()/(float)(RAND_MAX) - 0.5; // random float between -0.5 and 0.5;
		inputs[0] *= 100;
		inputs[0] = round(inputs[0]);
		inputs[0] /= 100;
		inputs[1] = (float)rand()/(float)(RAND_MAX) - 0.5; // random float between -0.5 and 0.5;
		inputs[1] *= 100;
		inputs[1] = round(inputs[1]);
		inputs[1] /= 100;
		float expected = inputs[0] + inputs[1];

		ann_run(inputs, outputs, my_ann);
		outputs[0] *= 100;
		outputs[0] = round(outputs[0]);
		outputs[0] /= 100;
		printf("%f + %f = %f. Expected output is %f\n", inputs[0], inputs[1], outputs[0], expected);
		if(outputs[0] == expected)
			score++;
		err_av += abs(expected - outputs[0]);
	}
	printf("ANN scored %d/%d, with an average error of %f\n", score, 100, err_av);

	return 0;
}
