# A simple Makefile

# List the object files in one place
# The first target is the default if you just say "make".  In this
# case, "build" relies on "sample", because I want the executable to be
# called "sample"

build:	
	nvcc primeV_stream.cu -o primeV_stream -arch=sm_30


# Before testing, we must compile.  
# Lines preceeded by @ aren't echoed before executing
# Execution will stop if a program has a non-zero return code;
# precede the line with a - to override that
test:	build
	./primeV_stream input.txt
	

.PHONY: exec
exec: build
