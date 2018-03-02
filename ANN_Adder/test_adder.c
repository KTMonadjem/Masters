/*
 * test_adder.c
 *
 *  Created on: Feb 26, 2018
 *      Author: keyan
 */

#include "test_adder.h"

#define TEST_SAMPLES 1000
#define INPUTS 2
#define OUTPUTS 1
#define DECIMALS 2
#define LAYERS 4

int main(int argc, char **argv)
{
	unsigned int i;
	srand(time(NULL));
	// if necessary, generate new test data
	char new_file[] = "new";
	if(argc > 1 && strcmp(argv[1], new_file) == 0)
	{
		printf("Training the ANN with a new set of random data.\n");
		printf("Generating 100 samples, with 2 inputs and 1 output each.\n");
		// Generating test data
		FILE *fp;
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

	fann_type *calc_out;
	int ret = 0;

	struct fann *ann;
	struct fann_train_data *test_data;

	printf("Creating network.\n");

	int filenum = 0;
	if(argc > 2)
	{
		filenum = argv[2][0] - 48;
	}
	char filename[11];
	printf("Opening ANN from file number: %d\n", filenum);
	sprintf(filename, "adder%d.net\0", filenum);
	ann = fann_create_from_file(filename);

	if(!ann)
	{
		printf("Error creating ann --- ABORTING.\n");
		return -1;
	}

	fann_print_connections(ann);
	fann_print_parameters(ann);

	printf("Testing network.\n");

	test_data = fann_read_train_from_file("test_adder.data");

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

	printf("Cleaning up.\n");
	fann_destroy_train(test_data);
	fann_destroy(ann);

	return 0;
}
