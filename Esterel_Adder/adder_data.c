/*
 * adder_data.c
 *
 *  Created on: Feb 26, 2018
 *      Author: keyan
 */

#include "adder.h"
#include "FANN/floatfann.c"

#define new_max(x,y) (abs(x) >= abs(y)) ? abs(x) : abs(y)
#define SAMPLES 1000
#define RANGE 100

int xval = 0;
int yval = 0;
int sum1 = 0;
int sum2 = 0;

void sendx(int num)
{
	xval = num;
	printf("x value sent: %d\n", xval);
}
void sendy(int num)
{
	yval = num;
	printf("y value sent: %d\n", yval);
}

int getx()
{
	printf("x value retrieved: %d\n", xval);
	return xval;
}
int gety()
{
	printf("y value retrieved: %d\n", yval);
	return yval;
}

void sendsum1(int num)
{
	sum1 = num;
	printf("sum1 value sent: %d\n", sum1);
}
void sendsum2(int num)
{
	sum2 = num;
	printf("sum2 value sent: %d\n", sum2);
}

int getsum1()
{
	printf("sum1 value retrieved: %d\n", sum1);
	return sum1;
}
int getsum2()
{
	printf("sum2 value retrieved: %d\n", sum2);
	return sum2;
}

void compare(char ** res, int sum1, int sum2)
{
	if(sum1 == sum2)
	{
		printf("The sum values are the same.\n");
		sprintf(&res[0], "The values are the same");
	}
	else
	{
		printf("The sum values are different.\n");
		sprintf(&res[0], "The values are different");
	}
}

int add(int x, int y, int adder)
{
	// ANN setup
	struct fann_train_data *data;
	struct fann *ann;
	fann_type *calc_out;

	char filename[11];
	printf("Opening ANN from file number: %d\n", adder);
	sprintf(filename, "adder%d.net\0", adder);
	ann = fann_create_from_file(filename);

	float new_x;
	float new_y;
	float multiplier;
	int sum;

	printf("x and y being added: %d + %d\n", x, y);

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
	sum = (int)calc_out[0];

	fann_destroy_train(data);
	fann_destroy(ann);

	printf("Result is: %d + %d = %d\n", x, y, sum);

	return sum;
}

