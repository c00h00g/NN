#pragma once

#include <vector.h>

namespace NN {


class Layer {
public:
    Layer();
    void add_layer();
    uint32_t get_node_size();
    //第几层layer
    uint32_t level;
private:
    //所有对nodes
    std::vector<Node> nodes;
    //层与层之间的matrix
    std::vector<std::vector<double> > mat;
};

}
