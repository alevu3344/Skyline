## Makefile per la generazione degli input per il progetto di High
## Performance Computing 2024/2025, corso di laurea in Ingegneria e
## Scienze Informatiche, Universita' di Bologna.

## Ultima modifica: 2024-12-05, Moreno Marzolla <moreno.marzolla(at)unibo.it>
NPOINTS:=100000

CFLAGS=-Wall -Wpedantic -std=c99

.PHONY: ALL

ALL: \
	circle-N1000-D2.in \
	test1-N$(NPOINTS)-D3.in \
	test2-N$(NPOINTS)-D4.in \
	test3-N$(NPOINTS)-D10.in \
	test4-N$(NPOINTS)-D8.in \
	test5-N$(NPOINTS)-D20.in \
	test6-N$(NPOINTS)-D50.in \
	test7-N$(NPOINTS)-D200.in \
	worst-N$(NPOINTS)-D10.in \
	weak_scaling_1_proc.in \
	weak_scaling_2_proc.in \
	weak_scaling_3_proc.in \
	weak_scaling_4_proc.in \
	weak_scaling_5_proc.in \
	weak_scaling_6_proc.in \
	weak_scaling_7_proc.in \
	weak_scaling_8_proc.in

## a small 2d circle
circle-N1000-D2.in:
	rbox 1000 s D2 W1 > $@

## surface of a 3d square
test1-N$(NPOINTS)-D3.in:
	rbox $(NPOINTS) W0 > $@

## surface of a 4d sphere
test2-N$(NPOINTS)-D4.in:
	rbox $(NPOINTS) s D4 > $@

## 10d diamond
test3-N$(NPOINTS)-D10.in:
	rbox $(NPOINTS) d D10 > $@

## 8d symplex
test4-N$(NPOINTS)-D8.in:
	rbox $(NPOINTS) y D8 > $@

## 20d sphere
test5-N$(NPOINTS)-D20.in:
	rbox $(NPOINTS) s D20 > $@

## 50d diamond
test6-N$(NPOINTS)-D50.in:
	rbox $(NPOINTS) d D50 > $@

## 200d diamond
test7-N$(NPOINTS)-D200.in:
	rbox $(NPOINTS) d D200 > $@

## worst-case scenario
worst-N$(NPOINTS)-D10.in: inputgen
	./inputgen $(NPOINTS) 10 > $@

## weak scaling 1 processor
weak_scaling_1_proc.in: inputgen
	./inputgen 50000 20 > $@

## weak scaling 1 processor
weak_scaling_2_proc.in: inputgen
	./inputgen 100000 20 > $@

## weak scaling 1 processor
weak_scaling_3_proc.in: inputgen
	./inputgen 150000 20 > $@

## weak scaling 1 processor
weak_scaling_4_proc.in: inputgen
	./inputgen 200000 20 > $@

## weak scaling 1 processor
weak_scaling_5_proc.in: inputgen
	./inputgen 250000 20 > $@

## weak scaling 1 processor
weak_scaling_6_proc.in: inputgen
	./inputgen 300000 20 > $@

## weak scaling 1 processor
weak_scaling_7_proc.in: inputgen
	./inputgen 350000 20 > $@

## weak scaling 1 processor
weak_scaling_8_proc.in: inputgen
	./inputgen 400000 20 > $@

clean:
	\rm -f *.in inputgen
