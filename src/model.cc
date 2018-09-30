#include <string.h>
#include <assert.h>

#include "model.h"
#include "layer.h"

namespace nn {

NN::NN() {
    nodes.clear();
    memset(mat, 0, sizeof(mat));
}

/**
 * @brief : 对nn网络添加一层
 * @param node_num : 每层节点个数
 * @param level : 记录nn的第几层
 * @param acti_fun_name : 激活函数
 **/
void NN::add_layer(uint32_t node_num,
                   uint32_t level,
                   std::string& acti_fun_name) {
    assert(node_num > 0);
    assert(level > 0);

    //前一层是输入
    if (level == 1) {
        add_first_layer(node_num, level, acti_fun_name);
    } else {
        
    }
}

/**
 * @brief : 添加第一层
 **/
void NN::add_first_layer(uint32_t node_num,
                         uint32_t level,
                         const std::string& acti_fun_name) {
    assert(node_num > 0);
    assert(level > 0);

    Layer layer;

    //初始化矩阵
    uint32_t row = node_num;
    uint32_t col = input_nodes.size();
    layer.init(row, col, level);

    //添加节点
    layer.add_nodes(node_num);

    //加入层
    layers.push_back(layer);
}

/**
 * @brief : 添加第一层
 **/
void NN::add_middle_layer(uint32_t node_num,
                          const std::string& acti_fun_name) {
    assert(node_num > 0);
    assert(level > 1);

    Layer layer;

    //初始化矩阵
    uint32_t row = node_num;
    uint32_t col = layers[layers.size() - 1].get_node_size();
    uint32_t level = layers[layers.size() - 1].level + 1;
    layer.init(row, col, level);

    //添加节点
    layer.add_nodes(node_num);

    //加入层
    layers.push_back(layer);
}

/**
 * @brief : 添加节点
 **/
void NN::add_nodes(uint32_t node_num,
                   const std::string& acti_fun_name) {
    for (uint32_t i = 0; i < node_num; ++i) {
        add_one_node(acti_fun_name);
    }
}

/**
 * @brief : 添加一个节点
 **/
void NN::add_one_node(const std::string& acti_fun_name) {
    Node node;
    nodes.push_back(node);
    if (acti_fun_name == "sigmoid") {
        node.activation = &sigmoid;
    } else if (acti_fun_name == "tanh") {
        node.activation = &tanh;
    }
}

/**
 * @brief : 前向传播
 **/
void NN::forward() {
}
 
}//end namespace 
