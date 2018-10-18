CXX=c++
CXXFLAGS=-std=c++11 -g -march=native -D_TRACE
RM=rm -f

mini_nn : utils.o activation.o layer.o model.o mini_nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^

mini_nn.o : mini_nn.cc
	$(CXX) $(CXXFLAGS) -c $<

model.o : model.cc model.h
	$(CXX) $(CXXFLAGS) -c $<

layer.o : layer.cc layer.h
	$(CXX) $(CXXFLAGS) -c $<

activation.o : activation.cc activation.h
	$(CXX) $(CXXFLAGS) -c $<

utils.o : utils.cc utils.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	$(RM) *.o mini_nn
