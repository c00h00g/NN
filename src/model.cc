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
 * @brief : ��nn�������һ��
 * @param node_num : ÿ��ڵ����
 * @param level : ��¼nn�ĵڼ���
 * @param acti_fun_name : �����
 **/
void NN::add_layer(uint32_t node_num,
                   uint32_t level,
                   std::string& acti_fun_name) {
    assert(node_num > 0);
    assert(level > 0);

    //ǰһ��������
    if (level == 1) {
        add_first_layer(node_num, level, acti_fun_name);
    } else {
        
    }
}

/**
 * @brief : ��ӵ�һ��
 **/
void NN::add_first_layer(uint32_t node_num,
                         uint32_t level,
                         const std::string& acti_fun_name) {
    assert(node_num > 0);
    assert(level > 0);

    Layer layer;

    //��ʼ������
    uint32_t row = node_num;
    uint32_t col = input_nodes.size();
    layer.init(row, col, level);

    //��ӽڵ�
    layer.add_nodes(node_num);

    //�����
    _layers.push_back(layer);
}

/**
 * @brief : ��ӵ�һ��
 **/
void NN::add_layer(uint32_t node_num,
                   const std::string& acti_fun_name) {
    assert(node_num > 0);
    assert(level > 1);

    //��ӵ�һ��
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
    //soft��ڵ�����������һ����ͬ
    if (acti_fun_name == "softmax") {
        assert(node_num == last_layer_nd_num);
    }

    uint32_t last_level = _layers[last_layer_idx].level;
    layer.init(row, last_layer_nd_num, last_level + 1);

    layer.add_nodes(node_num, acti_fun_name);

    _layers.push_back(layer);
}

/**
 * @brief :  ����loss�����һ����ݶ�
 **/
double NN::calc_last_layer_grad(Layer& last_layer,
                                const std::vector<double>& labels,
                                const std::string& loss_type) {
    if (loss_type == "cross-entropy") {
        
    } else if (loss_type == "squared-loss") {
        
    }
}

/**
 * @brief : ���㽻����loss
 **/
double NN::calc_cross_entropy_loss(Layer& last_layer,
                                   const std::vector<double>& labels,
                                   const std::string& loss_type) {
}

/**
 * @brief : ����ƽ��loss
 **/
double NN::calc_squared_loss(Layer& last_layer,
                             const std::vector<double>& labels,
                             const std::string& loss_type) {
}

/**
 * @brief : ǰ�򴫲�
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
 * @brief : ����һ���ڵ��ֵ
 **/
double NN::calc_node_value(const std::vector<double>& vec,
                           Node& node) {
    
}

void NN::other_layer_forward(Layer& left_layer,
                             Layer& right_layer) {
    
}
 
}//end namespace 
