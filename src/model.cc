#include "model.h"
#include "layer.h"

namespace NN {

MINI_NN::MINI_NN() {
    input_nodes.clear();
    _layers.clear();
}

void MINI_NN::add_loss_func(const std::string& loss_type) {
    _loss_type = loss_type;
}

void MINI_NN::fill_data(
    const std::vector<double>& train_line,
    const std::vector<uint32_t>& train_label) {
    
    input_nodes.assign(train_line.begin(), train_line.end());
    _labels.assign(train_label.begin(), train_label.end());
}

/**
 * @brief 
 * @param x_train : ѵ����������
 * @param y_train : ѵ��������Ӧ��lable
 * @param epochs : ѵ��������
 **/
void MINI_NN::fit(const std::vector<std::vector<double> >& x_train,
                  const std::vector<std::vector<uint32_t> >& y_train,
                  uint32_t epoch) {
    assert(x_train.size() == y_train.size());
    assert(epoch > 0);

    while (epoch--) {
        for (uint32_t i = 0; i < x_train.size(); ++i) {

            //feed����
            fill_data(x_train[i], y_train[i]);

            //ǰ�򴫲�
            forward();

            //���򴫲�
            backward();
        }
    }
}

/**
 * @brief : ��ӵ�һ��
 * @param node_num : �ڵ����
 **/
void MINI_NN::add_first_layer(
                     uint32_t node_num,
                     const std::string& acti_fun_name) {
    assert(node_num > 0);

    Layer layer;

    //��ʼ������
    uint32_t row = node_num;
    uint32_t col = input_nodes.size();
    layer.init(row, col, 1);

    //��ӽڵ�
    layer.add_nodes(node_num, acti_fun_name);
    layer._acti_type = acti_fun_name;

    //�����
    _layers.push_back(layer);
}

void MINI_NN::add_input_layer(uint32_t input_num,
                              uint32_t output_num) {
    input_nodes.resize(input_num);
}

/**
 * @brief : ��ӵ�һ��
 **/
void MINI_NN::add_layer(uint32_t node_num,
                        const std::string& acti_fun_name) {
    assert(node_num > 0);

    //��ӵ�һ��
    if (_layers.size() == 0) {
        add_first_layer(node_num, acti_fun_name);
    }else {
        add_middle_layer(node_num, acti_fun_name);
    }
}

void MINI_NN::add_middle_layer(uint32_t node_num,
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
    layer._acti_type = acti_fun_name;

    _layers.push_back(layer);
}

/**
 * @brief :  ����loss�����һ����ݶ�
 **/
void MINI_NN::calc_last_layer_grad(const std::vector<uint32_t>& labels) {
    if (_loss_type == "cross-entropy") {
        calc_cross_entropy_last_layer_grad(labels);
    } else if (_loss_type == "squared-loss") {
        calc_squared_last_layer_grad(labels);
    }
}

void MINI_NN::calc_middle_layer_grad() {
    int32_t last_layer_idx = -1;
    uint32_t layer_num = _layers.size();
    if (_loss_type == "cross-entropy") {
        //����softmax��
        last_layer_idx = layer_num - 2;
    } else if (_loss_type == "squared-loss") {
        last_layer_idx = layer_num - 1;
    }

    for (uint32_t i = last_layer_idx; i >= 0; i--) {
        //first layer
        if (i == 0) {
            Layer& first_layer = _layers[i];
            calc_first_layer_grad(first_layer);
        } else { // middle layer
            Layer& right_layer = _layers[i];
            Layer& left_layer = _layers[i - 1];
            calc_two_layer_grad(left_layer, right_layer);
        }
    }

    return;
}

void MINI_NN::calc_two_layer_grad(Layer& left_layer,
                                  Layer& right_layer) {
    //����ڵ��ݶ�
    std::vector<std::vector<double> >& mat = right_layer.mat; 
    std::vector<std::vector<double> >& grad = right_layer.grad; 
    for (uint32_t i = 0; i < left_layer.nodes.size(); ++i) {
        calc_one_node_backward(left_layer.nodes[i],
                               i,
                               right_layer,
                               mat);
    }

    //�ݶ��½�
    update_layer_weight(mat, grad);
}

void MINI_NN::update_layer_weight(std::vector<std::vector<double> >& mat,
                                  std::vector<std::vector<double> >& grad) {
    for (uint32_t i = 0; i < mat.size(); ++i) {
        for (uint32_t j = 0; j < mat[0].size(); ++j) {
            mat[i][j] -= grad[i][j];
        }
    }
}

/**
 * @brief : �����һ����ݶ� && �����ݶ�
 **/
void MINI_NN::calc_first_layer_grad(Layer& first_layer) {
    std::vector<std::vector<double> >& mat = first_layer.mat;
    std::vector<std::vector<double> >& grad = first_layer.grad;
    //�����ݶ�
    for (uint32_t i = 0; i < input_nodes.size(); ++i) {
        for (uint32_t j = 0; j < first_layer.nodes.size(); ++i) {
            grad[j][i] = first_layer.nodes[j].devi_b_value * input_nodes[i];
        }
    }

    //�����ݶ�
    update_layer_weight(mat, grad);
}

/**
 * @brief : ���㷴�򴫲�
 * @param node : ���һ��һ���ڵ�
 * @param node_idx : �ڵ�����
 * @param right_layer : �ұ�һ��
 * @param mat : left_layer & right_layer֮��ľ���
 **/
void MINI_NN::calc_one_node_backward(Node& node, 
                                     uint32_t node_idx,
                                     Layer& right_layer,
                                     std::vector<std::vector<double> >& mat) {
    double sum = 0;
    for (uint32_t i = 0; i < right_layer.nodes.size(); ++i) {
        sum += mat[i][node_idx] * right_layer.nodes[i].devi_b_value;

        //���������ݶ�
        right_layer.grad[i][node_idx] = right_layer.nodes[i].devi_b_value * node.a_value;
        //�����ݶ�?
    }
    node.devi_a_value = sum;
    node.devi_b_value = node.activation_devi(node.devi_a_value);
}

/**
 * @brief : ���򴫲�
 **/
void MINI_NN::backward() {
    calc_last_layer_grad(_labels);
    calc_middle_layer_grad();
}

/**
 * @brief : ���㽻����loss
 **/
double MINI_NN::
calc_cross_entropy_last_layer_grad(const std::vector<uint32_t>& labels) {
    uint32_t layer_num = _layers.size();
    assert(layer_num >= 2);
    Layer& softmax_layer = _layers[layer_num - 1];
    Layer& last_layer = _layers[layer_num - 2];
    for (uint32_t i = 0; i < last_layer.nodes.size(); ++i) {
        Node& node = last_layer.nodes[i];
        //softmax��ֻ��¼���յ�ֵa_value, ��softmaxֵ
        node.devi_a_value = 
            labels[i] == 1 ? softmax_layer.nodes[i].a_value - 1 : softmax_layer.nodes[i].a_value;
        node.devi_b_value = node.activation_devi(node.devi_a_value);
    }
}

/**
 * @brief : ����ƽ��loss
 **/
double MINI_NN::
calc_squared_last_layer_grad(const std::vector<uint32_t>& labels) {
    uint32_t layer_num = _layers.size();
    Layer& last_layer = _layers[layer_num - 1];
    for (uint32_t i = 0; i < last_layer.nodes.size(); ++i) {
        Node& node = last_layer.nodes[i];
        //����֮����
        node.devi_a_value = 
           labels[i] == 1 ? -1 * (1 - node.a_value) : node.a_value;
        //����֮ǰ����
        node.devi_b_value = node.activation_devi(node.devi_a_value);
    }
}

/**
 * @brief : ǰ�򴫲�
 **/
void MINI_NN::forward() {
    for (uint32_t i = 0; i < _layers.size(); ++i) {
        if (i == 0) {
            first_layer_forward(_layers[i]);
        }else {
            if (_layers[i]._acti_type == "softmax") {
                softmax_layer_forward(_layers[i - 1], _layers[i]);
            } else {
                middle_layer_forward(_layers[i - 1], _layers[i]);
            }
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
 * @brief : ����һ���ڵ��ֵ
 **/
void MINI_NN::calc_first_layer_node_forward(
             const std::vector<double>& weight,
             Node& node) {
    double sum = 0.0;
    for (uint32_t i = 0; i < weight.size(); ++i) {
        sum += weight[i] * input_nodes[i];
    }

    //������Ӻ��ֵ
    node.b_value = sum + node.b;

    //������ֵ
    node.a_value = node.activation(node.b_value);
}

/**
 * @����ǵ�һ��֮ǰ�����򴫲�
 * @param left_layer : ���һ��
 * @param right_layer : �ұ�һ��
 **/
void MINI_NN::middle_layer_forward(Layer& left_layer,
                                  Layer& right_layer) {
    std::vector<std::vector<double> >& mat = right_layer.mat;
    for (uint32_t i = 0; i < right_layer.nodes.size(); ++i) {
        calc_middle_layer_node_forward(mat[i], 
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
 * @brief : ����layer softmax֮��
 * @param layer : ��Ҫ��softmax�Ĳ�
 **/
double MINI_NN::softmax_sum(const Layer& layer) {
    double sum = 0.0;
    for (uint32_t i = 0; i < layer.nodes.size(); ++i) {
        sum += std::exp(layer.nodes[i].a_value);
    }
    return sum;
}

void MINI_NN::softmax_layer_forward(const Layer& left_layer,
                                    Layer& right_layer) {
    double sum = softmax_sum(left_layer);
    for (uint32_t i = 0; i < right_layer.nodes.size(); ++i) {
        right_layer.nodes[i].a_value = 
                      std::exp(left_layer.nodes[i].a_value) / sum;
    }
}

/**
 * @brief : ʹ��ǰһ�� & Ȩ�ؾ��������һ��Node��ֵ
 * @param weight : �ұ�layerһ���ڵ��Ӧ��Ȩ�ؾ���
 * @param left_layer
 * @param node : �ұ���Ҫ����Ľڵ�
 **/
void MINI_NN::calc_middle_layer_node_forward(
               const std::vector<double>& weight,
               Layer& left_layer,
               Node& node) {
    double sum = 0.0;
    for (uint32_t i = 0; i < weight.size(); ++i) {
        sum += weight[i] * left_layer.nodes[i].a_value;
    }
    node.b_value = sum + node.b;
    node.a_value = node.activation(node.b_value);
}

}//end namespace 
