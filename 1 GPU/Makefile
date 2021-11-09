TARGET=main
OBJECTS=timer.o util.o colorizer.o

CXXFLAGS=-std=c++11 -O3 -Wall -fopenmp
LDLIBS=-lm

# CUDA settings
CFLAGS+=-I/usr/local/cuda/include
LDFLAGS+=-L/usr/local/cuda/lib64
LDLIBS+=-lcudart -lstdc++

all: $(TARGET)

$(TARGET): $(OBJECTS)

colorizer.o: colorizer.cu
	nvcc -c -o $@ $^ -Xcompiler=-fopenmp

clean:
	rm -rf $(TARGET) $(OBJECTS)

performance:
	srun -N 1 ./main input.bin ../common/network.bin output.bin

	
test:
	srun -N 1 ./main ../common/bins/inputN1.bin ../common/network.bin output.bin

test8:
	srun -N 1 ./main ../common/bins/inputN8.bin ../common/network.bin output.bin
	
test32:
	srun -N 1 ./main ../common/bins/inputN32.bin ../common/network.bin output32.bin
	
compare:
	../common/compbin ./output.bin ../common/bins/answerN1.bin

compare32:
	../common/compbin ./output32.bin ../common/bins/answerN32.bin

nsys:
	srun nsys ./main ../common/bins/inputN1.bin ../common/network.bin output.bin