TARGET=main
OBJECTS=timer.o util.o colorizer.o

CXXFLAGS=-std=c++11 -O3 -Wall -fopenmp
LDLIBS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

performance:
	srun -N 1 ./main input.bin ../common/network.bin output.bin

	
test:
	srun -N 1 ./main ../common/bins/inputN1.bin ../common/network.bin output.bin

test8:
	srun -N 1 ./main ../common/bins/inputN8.bin ../common/network.bin output.bin
test32:
	srun -N 1 ./main ../common/bins/inputN32.bin ../common/network.bin output.bin
	
compare:
	../common/compbin ./output.bin ../common/bins/answerN1.bin

compare32:
	../common/compbin ./output.bin ../common/bins/answerN32.bin
