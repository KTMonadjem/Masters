/*
 * adder.h
 *
 *  Created on: Feb 26, 2018
 *      Author: keyan
 */

#ifndef ADDER_H_
#define ADDER_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int add(int, int, int);
void sendx(int);
void sendy(int);
int getx(void);
int gety(void);
void sendsum1(int num);
void sendsum2(int num);
int getsum1();
int getsum2();
void compare(char ** res, int sum1, int sum2);

#endif /* ADDER_H_ */
