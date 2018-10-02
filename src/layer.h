#pragma once

#include <vector.h>
#include "common.h"

namespace NN {

//激活函数
enum ACTI_FUNC {
    sigmoid,
    tanh,
    relu,
    softmax
};

class Layer {
public:
    Layer();

    void add_nodes();

    void NN::add_one_node(const std::string& acti_fun_name);

    uint32_t get_node_size();

    //第几层layer
    uint32_t level;

    //激活函数
    ACTI_FUNC _acti_func;
private:
    //所有对nodes
    std::vector<Node> nodes;
    //层与层之间的matrix
    std::vector<std::vector<double> > mat;
    //loss对weight的梯度
    std::vector<std::vector<double> > grad;
};

}
