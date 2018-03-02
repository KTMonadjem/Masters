/*
 * train_adder.c
 *
 *  Created on: Feb 26, 2018
 *      Author: keyan
 */

#include "train_adder.h"

#define TRAIN_SAMPLES 1000
#define TEST_SAMPLES 100
#define INPUTS 2
#define OUTPUTS 1
#define DECIMALS 2
#define LAYERS 4

int main(int argc, char **argv)
{
	srand(time(NULL));
	printf("This program trains the adder neural network\n");

	// First write the addition training data to a file in the range [-1, 1]
	char new_file[] = "new";
	if(argc > 1 && strcmp(argv[1], new_file) == 0)
	{
		printf("Training the ANN with a new set of random data.\n");
		printf("Generating 100 samples, with 2 inputs and 1 output each.\n");
		FILE *fp;
		fp = fopen("train_adder.data", "w");

		fprintf(fp, "%d %d %d\n", TRAIN_SAMPLES, INPUTS, OUTPUTS);

		int i;
		for(i = 0; i < TRAIN_SAMPLES; i++)
		{
			float x = (float)rand()/(float)(RAND_MAX) - 0.5;
			x = x * pow(10, DECIMALS);
			x = round(x);
			x = x/pow(10, DECIMALS);
			float y = (float)rand()/(float)(RAND_MAX) - 0.5;
			y = y * pow(10, DECIMALS);
			y = round(y);
			y = y/pow(10, DECIMALS);
			float sum = x + y;
			// printf("%f + %f = %f\n", x, y, sum);
			fprintf(fp, "%f %f\n%f\n", x, y, sum);
		}
		fclose(fp);

		// Generating test data
		fp = fopen("test_adder.data", "w");
		fprintf(fp, "%d %d %d\n", TEST_SAMPLES, INPUTS, OUTPUTS);

		for(i = 0; i < TEST_SAMPLES; i++)
		{
			float x = (float)rand()/(float)(RAND_MAX) - 0.5;
			x = x * pow(10, DECIMALS);
			x = round(x);
			x = x/pow(10, DECIMALS);
			float y = (float)rand()/(float)(RAND_MAX) - 0.5;
			y = y * pow(10, DECIMALS);
			y = round(y);
			y = y/pow(10, DECIMALS);
			float sum = x + y;
			// printf("%f + %f = %f\n", x, y, sum);
			fprintf(fp, "%f %f\n%f\n", x, y, sum);
		}
		fclose(fp);
	}

	// Setting up and training the ANN
	fann_type *calc_out;
	const unsigned int num_neurons_hidden_1 = 4;
	const unsigned int num_neurons_hidden_2 = 4;
	const float desired_error = (const float) 0;
	const unsigned int max_epochs = 1000;
	const unsigned int epochs_between_reports = 10;
	struct fann *ann;
	struct fann_train_data *train_data;
	struct fann_train_data *test_data;

	unsigned int i = 0;

	printf("Creating network.\n");
	ann = fann_create_standard(LAYERS, INPUTS, num_neurons_hidden_1, num_neurons_hidden_2, OUTPUTS);

	train_data = fann_read_train_from_file("train_adder.data");
	test_data = fann_read_train_from_file("test_adder.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_set_learning_rate(ann, 0.7);

	fann_init_weights(ann, train_data);

	printf("Training network.\n");
	fann_train_on_data(ann, train_data, max_epochs, epochs_between_reports, desired_error);

	printf("Testing network. %f\n", fann_test_data(ann, train_data));

	int count_incorrect = 0;
	int count_total = 0;
	for(i = 0; i < fann_length_train_data(test_data); i++)
	{
		calc_out = fann_run(ann, test_data->input[i]);
		calc_out[0] = calc_out[0] * pow(10, DECIMALS);
		calc_out[0] = round(calc_out[0]);
		calc_out[0] = calc_out[0]/pow(10, DECIMALS);
		printf("Addition test: %f + %f = %f; answer should be %f, error=%f\n",
				test_data->input[i][0], test_data->input[i][1], calc_out[0], test_data->output[i][0],
			   fann_abs(calc_out[0] - test_data->output[i][0]));
		if(fann_abs(calc_out[0] - test_data->output[i][0]) == 0)
			count_incorrect++;
		count_total++;
	}
	printf("The ANN scored %d/%d.\n", count_incorrect, count_total);

	printf("Saving network.\n");
	int filenum = 0;
	if(argc > 2)
	{
		filenum = argv[2][0] - 48;
	}
	char filename[11];
	printf("Saving ANN to file number: %d\n", filenum);
	sprintf(filename, "adder%d.net\0", filenum);
	fann_save(ann, filename);

//	int dec = fann_save_to_fixed(ann, "adder_fixed.net");
//	fann_save_train_to_fixed(test_data, "fixed.data", dec);

	printf("Cleaning up.\n");
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fann_destroy(ann);

	return 0;
}
