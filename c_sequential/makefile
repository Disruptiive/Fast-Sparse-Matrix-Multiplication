CC=gcc

Triangles: c_seq_triangles.o mmio.o
	$(CC)  -o Triangles c_seq_triangles.o mmio.o

c_seq_triangles.o:  c_seq_triangles.c
	$(CC) -c  c_seq_triangles.c

mmio.o:   mmio.c 
	$(CC) -c mmio.c

clean: 
	$(RM) count *.o *~