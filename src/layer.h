#pragma once

#include <vector.h>
#include "common.h"

namespace NN {

//�����
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

    //�ڼ���layer
    uint32_t level;

    //�����
    ACTI_FUNC _acti_func;
private:
    //���ж�nodes
    std::vector<Node> nodes;
    //�����֮���matrix
    std::vector<std::vector<double> > mat;
    //loss��weight���ݶ�
    std::vector<std::vector<double> > grad;
};

}
