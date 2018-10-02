#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <assert.h>

#include "layer.h"
#include "node.h"

namespace NN {

class MINI_NN {
public:
    MINI_NN();
    
    //正向传播
    void forward();

    //反向传播
    void backward();

    //计算其他层的梯度
    void calc_middle_layer_grad();

    void add_first_layer(uint32_t node_num,
                         const std::string& acti_fun_name);

    void add_middle_layer(uint32_t node_num,
                         const std::string& acti_fun_name);
    
    void add_layer(uint32_t node_num,
                   const std::string& acti_fun_name);

    void first_layer_forward(Layer& first_layer);

    void middle_layer_forward(Layer& left_layer,
                             Layer& right_layer);

    void calc_first_layer_node_forward(
                   const std::vector<double>& weight,
                   Node& node);

    void calc_middle_layer_node_forward(
                   const std::vector<double>& weight,
                   Layer& left_layer,
                   Node& node);

    double calc_last_layer_grad(const std::vector<double>& labels);

    double calc_cross_entropy_last_layer_grad(const std::vector<double>& labels);

    double calc_squared_last_layer_grad(const std::vector<double>& labels);

    void calc_two_layer_grad(Layer& left_layer,
                             Layer& right_layer);

    void calc_one_node_backward(Node& node, 
                                uint32_t node_idx,
                                Layer& right_layer,
                                std::vector<std::vector<double> >& mat);

    void softmax_layer_forward(const Layer& left_layer,
                               Layer& right_layer);

    double softmax_sum(const Layer& layer);

private:
    //loss函数类型
    std::string _loss_type;
private:
   //输入节点
   std::vector<double> input_nodes;

   //label值
   std::vector<double> _labels;

   //记录所有的层
   std::vector<Layer> _layers;
};
    
}
