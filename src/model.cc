#include <string>
#include <iostream>
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
                         const std::string& acti_fun_name) {
    assert(node_num > 0);

    Layer layer;

    //初始化矩阵
    uint32_t row = node_num;
    uint32_t col = input_nodes.size();
    layer.init(row, col, 1);

    //添加节点
    layer.add_nodes(node_num);

    //加入层
    _layers.push_back(layer);
}

/**
 * @brief : 添加第一层
 **/
void NN::add_layer(uint32_t node_num,
                   const std::string& acti_fun_name) {
    assert(node_num > 0);
    assert(level > 1);

    //添加第一层
    if (layer.size() == 0) {
        add_first_layer(node_num, 1, acti_fun_name);
    }else {
        add_other_layer(node_num, acti_fun_name);
    }
}

void NN::add_other_layer(uint32_t node_num,
                         const std::string& acti_fun_name) {
    Layer layer;

    uint32_t row = node_num;
    uint32_t last_layer_idx = _layers.size() - 1;
    uint32_t last_layer_nd_num = _layers[last_layer_idx].get_node_size();
    //soft层节点个数必须和上一层相同
    if (acti_fun_name == "softmax") {
        assert(node_num == last_layer_nd_num);
    }

    uint32_t last_level = _layers[last_layer_idx].level;
    layer.init(row, last_layer_nd_num, last_level + 1);

    layer.add_nodes(node_num, acti_fun_name);

    _layers.push_back(layer);
}

/**
 * @brief :  计算loss对最后一层的梯度
 **/
double NN::calc_last_layer_grad(Layer& last_layer,
                                const std::vector<double>& labels,
                                const std::string& loss_type) {
    if (loss_type == "cross-entropy") {
        
    } else if (loss_type == "squared-loss") {
        
    }
}

/**
 * @brief : 计算交叉熵loss
 **/
double NN::calc_cross_entropy_loss(Layer& last_layer,
                                   const std::vector<double>& labels,
                                   const std::string& loss_type) {
}

/**
 * @brief : 计算平方loss
 **/
double NN::calc_squared_loss(Layer& last_layer,
                             const std::vector<double>& labels,
                             const std::string& loss_type) {
}

/**
 * @brief : 前向传播
 **/
void NN::forward() {
    for (uint32_t i = 0; i < _layers.size(); ++i) {
        if (i == 0) {
            first_layer_forward(_layers[i]);
        }else {
            other_layer_forward(_layers[i - 1], _layers[i]);
        }
    }
}

void NN::first_layer_forward(Layer& first_layer) {
    std::vector<std::vector<double> >& mat = _layers[0].mat;
    for (uint32_t i = 0; i < first_layer.nodes.size(); ++i) {
        calc_first_layer_node_forward(mat[i], first_layer.nodes[i]);

#if _DEBUG
        std::cerr << "Layer idx : \n" << 0
                  << "Node idx : \n" << i
                  << "b_value : " << first_layer.nodes[i].b_value
                  << "a_value : " << first_layer.nodes[i].a_value;
#endif
    }
}

/**
 * @brief : 计算一个节点的值
 **/
void NN::calc_first_layer_node_forward(
             const std::vector<double>& weight,
             Node& node) {
    double sum = 0.0;
    for (uint32_t i = 0; i < weight.size(); ++i) {
        sum += weight[i] * input_nodes[i];
    }
    //线性相加后的值
    node.b_value = sum;
    //激活后的值
    node.a_value = node.activation(sum);
}

/**
 * @计算非第一层之前的正向传播
 * @param left_layer : 左边一层
 * @param right_layer : 右边一层
 **/
void NN::other_layer_forward(Layer& left_layer,
                             Layer& right_layer) {
    std::vector<std::vector<double> >& mat = right_layer.mat;
    for (uint32_t i = 0; i < right_layer.nodes.size(); ++i) {
        calc_other_layer_node_forward(mat[i], 
                                      left_layer,
                                      right_layer.nodes[i]);
#if _DEBUG
        std::cerr << "left Layer idx : " << left_layer.level
                  << "right Layer idx : " << right_layer.level
                  << "Node idx : \n" << i
                  << "b_value : " << right_layer.nodes[i].b_value
                  << "a_value : " << right_layer.nodes[i].a_value;
#endif
    }
}

/**
 * @brief : 使用前一层 & 权重矩阵更新新一层Node的值
 * @param weight : 右边layer一个节点对应的权重矩阵
 * @param left_layer
 * @param node : 右边需要计算的节点
 **/
void NN::calc_other_layer_node_forward(
               const std::vector<double>& weight,
               Layer& left_layer,
               Node& node) {
    double sum = 0.0;
    for (uint32_t i = 0; i < weight.size(); ++i) {
        sum += weight[i] * left_layer.nodes[i].a_value;
    }
    node.b_value = sum;
    node.a_value = node.activation(sum);
}
 
}//end namespace 
