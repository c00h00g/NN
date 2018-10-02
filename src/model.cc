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
        add_middle_layer();
    }
}

void NN::add_middle_layer(uint32_t node_num,
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

        }else {
            
        }
    }
}

void NN::first_layer_forward(Layer& first_layer) {
    for (uint32_t i = 0; i < first_layer.nodes.size(); ++i) {
        for (uint32_t j = 0; j < input_nodes.size(); ++j) {
            
        }
    }
}

/**
 * @brief : 计算一个节点的值
 **/
double NN::calc_node_value(const std::vector<double>& vec,
                           Node& node) {
    
}

void NN::other_layer_forward(Layer& left_layer,
                             Layer& right_layer) {
    
}
 
}//end namespace 
