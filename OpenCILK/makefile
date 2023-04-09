CC=gcc
CFLAGS= -fcilkplus

TrianglesCilk: cilk_triangles.o mmio.o
	$(CC) $(CFLAGS) -o TrianglesCilk cilk_triangles.o mmio.o

cilk_triangles.o:  cilk_triangles.o
	$(CC) $(CFLAGS) -c cilk_triangles.c

mmio.o:   mmio.c 
	$(CC) $(CFLAGS) -c mmio.c

clean: 
	$(RM) count *.o *~