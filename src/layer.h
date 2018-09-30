#pragma once

#include <vector.h>

namespace NN {


class Layer {
public:
    Layer();
    void add_layer();
    uint32_t get_node_size();
    //�ڼ���layer
    uint32_t level;
private:
    //���ж�nodes
    std::vector<Node> nodes;
    //�����֮���matrix
    std::vector<std::vector<double> > mat;
};

}
