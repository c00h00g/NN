#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <assert.h>
#include <map>
#include <limits>

#include "layer.h"
#include "node.h"
#include "utils.h"

namespace NN {

class MINI_NN {
public:
    //构造函数
    MINI_NN(const std::string& data_path,
            uint32_t epoch,
            double eta);

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

    void add_input_layer(uint32_t input_num);

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

    void fill_test_data(const std::vector<double>& line);

    template <typename T>
    std::string trans_vector_to_label(const std::vector<T>& fea_vec);

    void load_data(const std::string& data_path,
                   std::vector<std::vector<double> >& x_train,
                   std::vector<std::string>& y_train,
                   std::map<std::string, uint32_t>& _uniq_label_to_int,
                   std::map<uint32_t, std::string>& _uniq_int_to_label);

    void load_test_data(const std::string& path);

    uint32_t get_max_index(const std::vector<double>& value);

    void trans_labels(
           std::vector<std::string>& y_train,
           std::map<std::string, uint32_t>& uniq_labels);

    void fit();

    void predict(const std::string& path);

    void calc_softmax_loss();

private:
   //epoch
   uint32_t _epoch;

   //eta
   double _eta;

   //训练文件
   std::string _data_path;

   //loss函数类型
   std::string _loss_type;

   //输入节点
   std::vector<double> input_nodes;

   //label值
   std::vector<uint32_t> _labels;

   //记录所有的层
   std::vector<Layer> _layers;

   //x_train
   std::vector<std::vector<double> > _x_train;

   //y_train
   std::vector<std::vector<uint32_t> > _y_train;

   //y_train未转化为向量之前
   std::vector<std::string> _y_train_orig;

   //对lable进行编号
   std::map<std::string, uint32_t> _uniq_label_to_int;

   //将int转换为lable
   std::map<uint32_t, std::string> _uniq_int_to_label;

   //x_test
   std::vector<std::vector<double> > _x_test;

   //y_test未转换为向量之前
   std::vector<std::string> _y_test_orig;

   double _sum_loss;
   uint32_t _sum_loss_cnt;
};
    
}
