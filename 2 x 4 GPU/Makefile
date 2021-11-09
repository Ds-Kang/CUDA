TARGET=main
OBJECTS=timer.o util.o colorizer.o gpu_func.o

CXXFLAGS=-std=c++11 -O3 -Wall -fopenmp -DUSE_MPI 
LDLIBS=-lm
CXX=mpic++

# CUDA settings
OBJECTS+=gpu_func.o
CFLAGS+=-I/usr/local/cuda/include 
LDFLAGS+=-L/usr/local/cuda/lib64 
LDLIBS+=-lcudart -lstdc++

all: $(TARGET)

$(TARGET): $(OBJECTS)

gpu_func.o: gpu_func.cu
	nvcc -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

performance:
	salloc -N 2 mpirun -npernode 4 --oversubscribe ./main input.bin ../common/network.bin output.bin

	
test:
	salloc -N 2 mpirun -npernode 4  --oversubscribe ./main ../common/bins/inputN8.bin ../common/network.bin output8.bin
	

	
test32:
	salloc -N 2 mpirun -npernode 4  --oversubscribe ./main ../common/bins/inputN32.bin ../common/network.bin output32.bin
	
compare:
	../common/compbin ./output.bin ../common/bins/answerN1.bin
compare8:
	../common/compbin ./output8.bin ../common/bins/answerN8.bin

compare32:
	../common/compbin ./output32.bin ../common/bins/answerN32.bin

nsys:
	srun nsys ./main ../common/bins/inputN1.bin ../common/network.bin output.bin