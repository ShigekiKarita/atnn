INCPATH := -I$(ATEN_ROOT)/include -I..
LIBPATH := -L$(ATEN_ROOT)/lib
LIBS := -lATen -lTH -lTHC -lTHS -lTHCS -lTHNN -lTHCUNN
CXX_FLAGS := -std=c++14 -g -Wall -Wextra -D_GLIBCXX_DEBUG

MAKE_PID := $(shell echo $$PPID)
JOB_FLAG := $(filter -j%, $(subst -j ,-j,$(shell ps T | grep "^\s*$(MAKE_PID).*$(MAKE)")))
JOBS     := $(subst -j,,$(JOB_FLAG))
ifndef JOBS
JOBS=5
endif

ifdef USE_BOOST
BOOST_LIB := -ldl -lboost_stacktrace_addr2line
BOOST_FLAGS := -DBOOST_STACKTRACE_USE_ADDR2LINE -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER -DHAVE_BOOST
else
BOOST_LIB :=
BOOST_FLAGS :=
endif

.PHONY: test clean

%.out: %.cpp
	g++ -o $@ $< $(CXX_FLAGS) $(BOOST_FLAGS) $(INCPATH) $(LIBPATH) $(LIBS) $(BOOST_LIB)

test: test_autograd.out test_variable.out test_nn.out
	find . -name "*.out" | xargs -n1 -P$(JOBS) sh -c

clean:
	rm -fv *.out

