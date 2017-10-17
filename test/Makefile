INCPATH := -I$(ATEN_ROOT)/include -I..
LIBPATH := -L$(ATEN_ROOT)/lib
LIBS := -ldl -lboost_stacktrace_addr2line  -lATen -lTH -lTHC -lTHS -lTHCS -lTHNN -lTHCUNN

.PHONY: test clean

%.out: %.cpp
	g++ -DBOOST_STACKTRACE_USE_ADDR2LINE -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER -o $@ $< -std=c++14 -g $(INCPATH) $(LIBPATH) $(LIBS)

test: test_autograd.out test_nn.out
	find . -name "*.out" -exec {} \;

clean:
	rm -fv *.out

