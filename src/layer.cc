#include "layer.h"

namespace NN {

Layer::Layer() {
    nodes.clear();
    mat.clear();
    grad.clear();
}

/**
 * @brief : layer�ڵ����
 **/
uint32_t Layer::get_node_size() {
    return nodes.size();
}

/**
 * @brief : ��ʼ����֮ǰ��mat, ��ʼ��ֵΪ0
 **/
void Layer::init(uint32_t m, 
                 uint32_t n,
                 uint32_t level) {
    assert(m > 0 && n > 0 && level >= 0);

    //��ʼ��Ȩ�ؾ���
    mat.resize(m);
    for (uint32_t i = 0; i < m; ++i) {
        mat[i].resize(n, 0);
    }

    //��ʼ���ݶȾ���
    grad.resize(m);
    for (uint32_t i = 0; i < m; ++i) {
        grad[i].resize(n, 0);
    }

    this->level = level;
}

/**
 * @brief : ��ӽڵ�
 **/
void Layer::add_nodes(uint32_t node_num,
                      const std::string& acti_fun_name) {
    for (uint32_t i = 0; i < node_num; ++i) {
        add_one_node(acti_fun_name);
    }
}

/**
 * @brief : ���һ���ڵ�
 **/
void Layer::add_one_node(const std::string& acti_fun_name) {
    Node node;
    if (acti_fun_name == "sigmoid") {
        node.activation = &nn_sigmoid;
        node.activation_devi = &nn_sigmoid_deri;
    } else if (acti_fun_name == "tanh") {
        node.activation = &nn_tanh;
        node.activation_devi = &nn_tanh_deri;
    } else if (acti_fun_name == "relu") {
        node.activation = &nn_relu;
        node.activation_devi = &nn_relu_deri;
    }
    nodes.push_back(node);
}

}
