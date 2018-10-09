#include "activation.h"

namespace NN {

double nn_log(double x) {
    return std::log(x + 1e-5);
}

double nn_sigmoid(double x) {
    return 1.0 / (1 + std::exp(-1.0 * x));
}

double nn_tanh(double x) {
    double a = std::exp(x) - std::exp(-1 * x);
    double b = std::exp(x) + std::exp(-1 * x);
    return a * 1.0 / b;
}

double nn_relu(double x) {
    return x > 0 ? x : 0;
}

/**
 * @brief : sigmoid导数
 **/
double nn_sigmoid_deri(double x) {
    double sig_value = nn_sigmoid(x);
    return sig_value * (1.0 - sig_value);
}

/**
 * @brief : tanh导数
 **/
double nn_tanh_deri(double x) {
    double tanh_value = nn_tanh(x);
    return 1 - tanh_value * tanh_value;
}

/**
 * @brief : 不是连续可导为什么可以用
 **/
double nn_relu_deri(double x) {
    return x > 0 ? 1 : 0;
}
    
};
