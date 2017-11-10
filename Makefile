OBJS = main.o gaussian.o
CC = g++
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

main : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o main

main.o : main.cpp gaussian.h
	$(CC) $(CFLAGS) main.cpp

gaussian.o : gaussian.cpp gaussian.h
	$(CC) $(CFLAGS) gaussian.cpp

clean:
	\rm *.o *~ main
