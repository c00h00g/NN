#include "layer.h"

namespace NN {

/**
 * @brief : layer节点个数
 **/
uint32_t Layer::get_node_size() {
    return nodes.size();
}

/**
 * @brief : 初始化层之前的mat, 初始化值为0
 **/
void Layer::init(uint32_t m, 
                 uint32_t n,
                 uint32_t level) {
    assert(m > 0 && n > 0 && level >= 0);

    //初始化权重矩阵
    mat.resize(m);
    for (uint32_t i = 0; i < m; ++i) {
        mat[i].resize(n, 0);
    }

    this->level = level;
}

/**
 * @brief : 添加节点
 **/
void Layer::add_nodes(uint32_t node_num,
                      const std::string& acti_fun_name) {
    for (uint32_t i = 0; i < node_num; ++i) {
        add_one_node(acti_fun_name);
    }
}

/**
 * @brief : 添加一个节点
 **/
void Layer::add_one_node(const std::string& acti_fun_name) {
    Node node;
    nodes.push_back(node);
    if (acti_fun_name == "sigmoid") {
        node.activation = &nn_sigmoid;
    } else if (acti_fun_name == "tanh") {
        node.activation = &nn_tanh;
    }
}

}
