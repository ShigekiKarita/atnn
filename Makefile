INCPATH := /home/skarita/tool/ATen/usr/include
LIBPATH := /home/skarita/tool/ATen/usr/lib
LIBS := -lATen -lTH -lTHC -lTHS -lTHCS -lTHNN -lTHCUNN

.PHONY: test

test:
	g++	-I$(INCPATH) -L$(LIBPATH) $(LIBS) -o test test.cpp -std=c++14 -g
	LD_LIBRARY_PATH=$(LIBPATH) ./test

