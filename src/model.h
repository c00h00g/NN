#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <assert.h>
#include <map>

#include "layer.h"
#include "node.h"
#include "utils.h"

namespace NN {

class MINI_NN {
public:
    //���캯��
    MINI_NN(const std::string& data_path,
            uint32_t epoch);

    //���򴫲�
    void forward();

    //���򴫲�
    void backward();

    //������������ݶ�
    void calc_middle_layer_grad();

    void add_first_layer(uint32_t node_num,
                         const std::string& acti_fun_name);

    void add_middle_layer(uint32_t node_num,
                         const std::string& acti_fun_name);
    
    void add_layer(uint32_t node_num,
                   const std::string& acti_fun_name);

    void add_input_layer(uint32_t input_num, uint32_t output_num);

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

    void calc_last_layer_grad(const std::vector<uint32_t>& labels);

    double calc_cross_entropy_last_layer_grad(const std::vector<uint32_t>& labels);

    double calc_squared_last_layer_grad(const std::vector<uint32_t>& labels);

    void calc_two_layer_grad(Layer& left_layer,
                             Layer& right_layer);

    void calc_one_node_backward(Node& node, 
                                uint32_t node_idx,
                                Layer& right_layer,
                                std::vector<std::vector<double> >& mat);

    void softmax_layer_forward(const Layer& left_layer,
                               Layer& right_layer);

    double softmax_sum(const Layer& layer);

    void update_layer_weight(std::vector<std::vector<double> >& mat,
                             std::vector<std::vector<double> >& grad);

    void calc_first_layer_grad(Layer& first_layer);

    void add_loss_func(const std::string& loss_type);

    void fill_data(const std::vector<double>& train_line,
                   const std::vector<uint32_t>& train_label);

    void load_data(const std::string& data_path,
                   std::vector<std::vector<double> >& x_train,
                   std::vector<std::string>& y_train,
                   std::map<std::string, uint32_t>& all_labels);

    void fit();

private:
   //epoch
   uint32_t _epoch;

   //ѵ���ļ�
   std::string _data_path;

   //loss��������
   std::string _loss_type;

   //����ڵ�
   std::vector<double> input_nodes;

   //labelֵ
   std::vector<uint32_t> _labels;

   //��¼���еĲ�
   std::vector<Layer> _layers;

   //x_train
   std::vector<std::vector<double> > _x_train;

   //y_train
   std::vector<std::vector<uint32_t> > _y_train;

   //y_trainδת��Ϊ����֮ǰ
   std::vector<std::string> _y_train_orig;

   //��lable���б��
   std::map<std::string, uint32_t> _uniq_labels;
};
    
}
