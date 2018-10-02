#pragma once

#include <vector>
#include <string>
#include <vector>
#include <assert.h>
#include <memory.h>

#include "activation.h"
#include "node.h"

namespace NN {

//�����
//enum ACTI_FUNC {
//    sigmoid,
//    tanh,
//    relu,
//    softmax
//};

class Layer {
public:
    Layer();

    void add_nodes();

    void add_one_node(const std::string& acti_fun_name);

    uint32_t get_node_size();

    void init(uint32_t m, 
              uint32_t n,
              uint32_t level);

    void add_nodes(uint32_t node_num,
                   const std::string& acti_fun_name);

public:
    //�ڼ���layer
    uint32_t level;

    //�����֮���matrix
    std::vector<std::vector<double> > mat;

    //loss��weight���ݶ�
    std::vector<std::vector<double> > grad;

    //���ж�nodes
    std::vector<Node> nodes;

    //loss��������
    std::string _acti_type;

    //�����
    //ACTI_FUNC _acti_func;
};

}
