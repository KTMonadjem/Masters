/*
 * main.c
 *
 *  Created on: Mar 5, 2018
 *      Author: keyan
 */

#include "ann.c"

int main(int argc, char **argv)
{
	int num_layers = 3;
	int layers[3] = {2, 3, 1};
	int max_weights = 6;
	float weights[2][6] = {{0.8, 0.2, 0.4, 0.9, 0.3, 0.5}, {0.3, 0.5, 0.9, 0, 0, 0}};
	int bias = 0;
	int activation = 0;
	struct ANN new_ann;
	struct ANN *my_ann = &new_ann;
	ann_init_custom(my_ann, num_layers, layers, max_weights, weights, bias, activation);

	float inputs[2] = {1, 1};
	float outputs[1];
	ann_run(inputs, outputs, my_ann);

	ann_print(my_ann, inputs, 0);
	//printf("The ANN took in %f and %f and returned %f.\n", inputs[0], inputs[1], outputs[0]);

	ann_train(my_ann, "test.data", 100, 0.01);

	ann_print(my_ann, inputs, 0);

	return 0;
}
