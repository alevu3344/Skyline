EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_SERIAL:=skyline
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) 
CFLAGS+=-std=c99 -Wall -Wpedantic -O2 -D_XOPEN_SOURCE=600
LDLIBS+=-lm
.PHONY: clean

ALL: $(EXE)


$(EXE_OMP): CFLAGS+=-fopenmp
openmp: $(EXE_OMP)

$(EXE_MPI): CC=mpicc
mpi: $(EXE_MPI)

clean:
	\rm -f $(EXE) *.o *~ *out *json *txt