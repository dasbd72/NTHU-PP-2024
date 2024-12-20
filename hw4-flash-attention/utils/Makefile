CC = gcc
CXX = g++

CFLAGS = -std=c++11 -O3 -Wall
LDFLAGS = -lm

TARGETS = diff

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean

diff: diff.cc
	$(CXX) $(CFLAGS) -o $@ $? $(LDFLAGS)
