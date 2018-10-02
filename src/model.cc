#include "model.h"
#include "layer.h"

namespace NN {

MINI_NN::MINI_NN() {
    input_nodes.clear();
}

/**
 * @brief : 添加第一层
 **/
void MINI_NN::add_first_layer(
                     uint32_t node_num,
                     const std::string& acti_fun_name) {
    assert(node_num > 0);

    Layer layer;

    //初始化矩阵
    uint32_t row = node_num;
    uint32_t col = input_nodes.size();
    layer.init(row, col, 1);

    //添加节点
    layer.add_nodes(node_num, acti_fun_name);

    //加入层
    _layers.push_back(layer);
}

/**
 * @brief : 添加第一层
 **/
void MINI_NN::add_layer(uint32_t node_num,
                        const std::string& acti_fun_name) {
    assert(node_num > 0);

    //添加第一层
    if (_layers.size() == 0) {
        add_first_layer(node_num, acti_fun_name);
    }else {
        add_other_layer(node_num, acti_fun_name);
    }
}

void MINI_NN::add_other_layer(uint32_t node_num,
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
double MINI_NN::calc_last_layer_grad(Layer& last_layer,
                                const std::vector<double>& labels,
                                const std::string& loss_type) {
    if (loss_type == "cross-entropy") {
        
    } else if (loss_type == "squared-loss") {
        
    }

    return 0.0;
}

/**
 * @brief : 计算交叉熵loss
 **/
double MINI_NN::calc_cross_entropy_loss(Layer& last_layer,
                                   const std::vector<double>& labels,
                                   const std::string& loss_type) {
}

/**
 * @brief : 计算平方loss
 **/
double MINI_NN::calc_squared_loss(Layer& last_layer,
                             const std::vector<double>& labels,
                             const std::string& loss_type) {
}

/**
 * @brief : 前向传播
 **/
void MINI_NN::forward() {
    for (uint32_t i = 0; i < _layers.size(); ++i) {
        if (i == 0) {
            first_layer_forward(_layers[i]);
        }else {
            other_layer_forward(_layers[i - 1], _layers[i]);
        }
    }
}

void MINI_NN::first_layer_forward(Layer& first_layer) {
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
void MINI_NN::calc_first_layer_node_forward(
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
void MINI_NN::other_layer_forward(Layer& left_layer,
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
void MINI_NN::calc_other_layer_node_forward(
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
