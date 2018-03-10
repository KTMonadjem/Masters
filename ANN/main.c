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


	// ANN ADDER
//	int num_layers = 4;
//	int layers[4] = {2, 4, 4, 1};
//	int bias = 1;
//	int activation = 1;
//	struct ANN new_ann;
//	struct ANN *my_ann = &new_ann;
//	ann_init(my_ann, num_layers, layers, bias, activation);
//
//	float inputs[2] = {1, 1};
//	float outputs[1];
//	//ann_run(inputs, outputs, my_ann);
//
//	ann_print(my_ann, inputs, 1);
//	//printf("The ANN took in %f and %f and returned %f.\n", inputs[0], inputs[1], outputs[0]);
//
//	ann_train_batch(my_ann, "train.data", 1000, 0.0001);
//
//	//ann_print(my_ann, inputs, 1);
//
//	int i = 0;
//	int score = 0;
//	float err_av = 0;
//	for(; i < 100; i++)
//	{
//		inputs[0] = (float)rand()/(float)(RAND_MAX) - 0.5; // random float between -0.5 and 0.5;
//		inputs[0] *= 100;
//		inputs[0] = round(inputs[0]);
//		inputs[0] /= 100;
//		inputs[1] = (float)rand()/(float)(RAND_MAX) - 0.5; // random float between -0.5 and 0.5;
//		inputs[1] *= 100;
//		inputs[1] = round(inputs[1]);
//		inputs[1] /= 100;
//		float expected = inputs[0] + inputs[1];
//
//		ann_run(inputs, outputs, my_ann);
//		outputs[0] *= 100;
//		outputs[0] = round(outputs[0]);
//		outputs[0] /= 100;
//		printf("%f + %f = %f. Expected output is %f\n", inputs[0], inputs[1], outputs[0], expected);
//		if(outputs[0] == expected)
//			score++;
//		err_av += (sqrt(pow(expected - outputs[0], 2)));
//	}
//	printf("ANN scored %d/%d, with an average error of %f\n", score, 100, err_av/100);

	// ANN comparator
//	int num_layers = 3;
//	int layers[3] = {2, 2, 1};
//	int bias = 1;
//	int activation = 1;
//	struct ANN new_ann;
//	struct ANN *my_ann = &new_ann;
//	ann_init(my_ann, num_layers, layers, bias, activation);
//	int i = 0;
//	int samples = 10000;
//	float inputs[2];
//	float outputs[1];
//	int score = 0;
//	float err_av = 0;
//
//	// train data
//	FILE *fp;
//	fp = fopen("train_comp.data", "w");
//	for(i = 0; i < samples; i++)
//	{
//		if(i == 0)
//			fprintf(fp, "%d\n", samples);
//
//		inputs[0] = (float)rand()/(float)(RAND_MAX/2) - 1;
//		inputs[0] *= 100;
//		inputs[0] = round(inputs[0]);
//		inputs[0] /= 100;
//
//		inputs[1] = (float)rand()/(float)(RAND_MAX/2) - 1;
//		inputs[1] *= 100;
//		inputs[1] = round(inputs[1]);
//		inputs[1] /= 100;
//
//		float expected = 0;
//		if(inputs[0] > 0 || inputs[1] > 0)
//			expected = -1;
//		else
//			expected = 1;
//
//		fprintf(fp, "%f %f\n%f\n", inputs[0], inputs[1], expected);
//	}
//	fclose(fp);
//
//	//ann_run(inputs, outputs, my_ann);
//
//	//ann_print(my_ann, inputs, 1);
//	inputs[0] = 1;
//	inputs[1] = 1;
//	ann_print(my_ann, inputs, 1);
//	//printf("The ANN took in %f and %f and returned %f.\n", inputs[0], inputs[1], outputs[0]);
//
//	ann_train_batch(my_ann, "train_comp.data", 500, 0.0001);
//
//	//ann_print(my_ann, inputs, 1);
//
//	// test data
//	for(i = 0; i < 100; i++)
//	{
//		inputs[0] = (float)rand()/(float)(RAND_MAX/2) - 1;
//		inputs[0] *= 100;
//		inputs[0] = round(inputs[0]);
//		inputs[0] /= 100;
//
//		inputs[1] = (float)rand()/(float)(RAND_MAX/2) - 1;
//		inputs[1] *= 100;
//		inputs[1] = round(inputs[1]);
//		inputs[1] /= 100;
//
//		float expected = 0;
//		if(inputs[0] > 0 || inputs[1] > 0)
//			expected = -1;
//		else
//			expected = 1;
//
//		ann_run(inputs, outputs, my_ann);
//
//		printf("%f and %f -> %f. Expected output is %f\n", inputs[0], inputs[1], outputs[0], expected);
//		if((inputs[0] < 0 && inputs[1] < 0 && outputs[0] > 0) || ((inputs[0] > 0 || inputs[1] > 0) && outputs[0] < 0))
//			score++;
//		err_av += (sqrt(pow(expected - outputs[0], 2)));
//	}
//	printf("ANN scored %d/%d, with an average error of %f\n", score, 100, err_av/100);

//	inputs[0] = 1;
//	inputs[1] = 1;
//	ann_print(my_ann, inputs, 0);

	return 0;
}
