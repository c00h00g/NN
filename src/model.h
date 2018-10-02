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
    
    //���򴫲�
    void forward();

    //���򴫲�
    void backward();

    //�������loss�����һ��������ݶ�
    double calc_last_layer_grad(Layer& last_layer,
                                const std::vector<double>& labels,
                                const std::string& loss_type);

    //������������ݶ�
    double calc_other_layer_grad();

    void add_first_layer(uint32_t node_num,
                         const std::string& acti_fun_name);

    void add_other_layer(uint32_t node_num,
                         const std::string& acti_fun_name);
    
    void add_layer(uint32_t node_num,
                   const std::string& acti_fun_name);

    double calc_cross_entropy_loss(Layer& last_layer,
                                   const std::vector<double>& labels,
                                   const std::string& loss_type);

    double calc_squared_loss(Layer& last_layer,
                             const std::vector<double>& labels,
                             const std::string& loss_type);

    void first_layer_forward(Layer& first_layer);

    void other_layer_forward(Layer& left_layer,
                             Layer& right_layer);

    void calc_first_layer_node_forward(
                   const std::vector<double>& weight,
                   Node& node);

    void calc_other_layer_node_forward(
                   const std::vector<double>& weight,
                   Layer& left_layer,
                   Node& node);

    //loss��������
    std::string _loss_type;
private:
   //����ڵ�
   std::vector<double> input_nodes;

   //labelֵ
   std::vector<double> _labels;

   //��¼���еĲ�
   std::vector<Layer> _layers;
};
    
}
