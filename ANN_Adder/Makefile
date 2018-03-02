goal: adder train_adder test_adder test_adder

adder: adder_data.c adder.h
	gcc adder_data.c -o adder -lfloatfann -lm
	
train_adder: train_adder.c
	gcc train_adder.c -o train_adder -lfloatfann -lm
	
test_adder: test_adder.c
	gcc test_adder.c -o test_adder -lfloatfann -lm
	
clean:
	-rm -f adder.o
	-rm -f adder
	-rm -f train_adder.o
	-rm -f train_adder
	-rm -f test_adder.o
	-rm -f test_adder