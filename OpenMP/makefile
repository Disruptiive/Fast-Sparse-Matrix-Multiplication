CC=gcc
CFLAGS= -fopenmp

TrianglesOMP: ompTriangles.o mmio.o
	$(CC) $(CFLAGS) -o TrianglesOMP ompTriangles.o mmio.o

ompTriangles.o:  ompTriangles.o
	$(CC) $(CFLAGS) -c ompTriangles.c

mmio.o:   mmio.c 
	$(CC) $(CFLAGS) -c mmio.c

clean: 
	$(RM) count *.o *~