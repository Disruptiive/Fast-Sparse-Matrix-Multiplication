CC=gcc
CFLAGS= -pthread

PthreadTriangle: pthreadTriangles.o mmio.o
	$(CC) $(CFLAGS) -o PthreadTriangle pthreadTriangles.o mmio.o

pthreadTriangles.o:  pthreadTriangles.o
	$(CC) $(CFLAGS) -c pthreadTriangles.c

mmio.o:   mmio.c 
	$(CC) $(CFLAGS) -c mmio.c

clean: 
	$(RM) count *.o *~