CXX=c++
CXXFLAGS=-std=c++11 -g -march=native
RM=rm -f

activation.o : activation.cc activation.h
	$(CXX) $(CXXFLAGS) -c $<

node.o : node.cc node.h
	$(CXX) $(CXXFLAGS) -c $<

layer.o : layer.cc layer.h
	$(CXX) $(CXXFLAGS) -c $<

model.o : model.cc model.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	$(RM) *.o
