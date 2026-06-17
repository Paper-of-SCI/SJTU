NVCC ?= nvcc
NVCCFLAGS ?= -O2

TARGETS := add_two_numbers add_many_numbers

.PHONY: all run clean

all: $(TARGETS)

add_two_numbers: 00_add_two_numbers.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

add_many_numbers: 01_add_many_numbers.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: all
	./add_two_numbers
	./add_many_numbers

clean:
	rm -f $(TARGETS)
