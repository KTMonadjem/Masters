/*
 * adder_data.c
 *
 *  Created on: Feb 26, 2018
 *      Author: keyan
 */

#include "adder.h"
#define new_max(x,y) (abs(x) >= abs(y)) ? abs(x) : abs(y)
#define SAMPLES 1000
#define RANGE 100

int add(int x, int y)
{
	// ANN setup
	int i = 0;
	struct fann_train_data *data;
	struct fann *ann;
	fann_type *calc_out;
	ann = fann_create_from_file("adder.net");

	float new_x;
	float new_y;
	float multiplier;

	// converting data to range [-0.5, 0.5]
	multiplier = new_max(x, y);
	multiplier *= 2;
	new_x = ((float)x)/multiplier;
	new_y = ((float)y)/multiplier;

	// writing data to data file
	FILE *fp;
	fp = fopen("adder.data", "w");
	fprintf(fp, "1 2 1\n%f %f\n%f\n", new_x, new_y, new_x + new_y);
	fclose(fp);

	// passing value through ANN
	data = fann_read_train_from_file("adder.data");

	calc_out = fann_run(ann, data->input[0]);
	calc_out[0] *= multiplier;
	calc_out[0] = round(calc_out[0]);

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);

	return (int)calc_out[0];
}

int main()
{
	srand(time(NULL));
	int i = 0;
	struct fann_train_data *data;
	struct fann *ann;
	fann_type *calc_out;

	printf("Adder is loading the ANN.\n");
	ann = fann_create_from_file("adder.net");
	if(!ann)
	{
		printf("Error creating ann --- ABORTING.\n");
		return -1;
	}

	fann_print_connections(ann);
	fann_print_parameters(ann);

	printf("Adder is now running.\n");

	int x;
	int y;
	int sum;
	float new_x;
	float new_y;
	float multiplier;
	int count_incorrect = 0;
	int count_total = 0;
	int error_sum = 0;
	for(i = 0; i < SAMPLES; i++)
	{
		x = rand()%(2*RANGE) - RANGE;
		y = rand()%(2*RANGE) - RANGE;
		sum = x + y;

		// converting data to range [-0.5, 0.5]
		multiplier = new_max(x, y);
		multiplier *= 2;
//		printf("x: %d\n", x);
//		printf("y: %d\n", y);
//		printf("Multiplier: %f\n", multiplier);
		new_x = ((float)x)/multiplier;
		new_y = ((float)y)/multiplier;

		// writing data to data file
		FILE *fp;
		fp = fopen("adder.data", "w");
		fprintf(fp, "1 2 1\n%f %f\n%f\n", new_x, new_y, new_x + new_y);
		fclose(fp);

		// passing value through ANN
		data = fann_read_train_from_file("adder.data");

		calc_out = fann_run(ann, data->input[0]);
		calc_out[0] *= multiplier;
		calc_out[0] = round(calc_out[0]);
		printf("Addition test: %d + %d = %d; answer should be %d, error=%d\n", x, y, (int)calc_out[0], sum, (int)fann_abs(calc_out[0] - sum));

		error_sum += (int)fann_abs(calc_out[0] - sum);
		if(fann_abs(calc_out[0] - sum) == 0)
			count_incorrect++;
		count_total++;
	}
	printf("The ANN scored %d/%d.\n", count_incorrect, count_total);
	printf("The average error across this ANN run was: %f%%\n", (float)error_sum/SAMPLES);

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);

	return 0;
}
