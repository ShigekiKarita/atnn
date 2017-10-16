INCPATH := -I$(ATEN_ROOT)/include -I..
LIBPATH := -L$(ATEN_ROOT)/lib
LIBS := -lATen -lTH -lTHC -lTHS -lTHCS -lTHNN -lTHCUNN

.PHONY: test clean

%.out: %.cpp
	g++ -o $@ $< -std=c++14 -g $(INCPATH) $(LIBPATH) $(LIBS)

test: test_autograd.out
	find -name *.out -exec {} \;

clean:
	rm -fv *.out

