#include "activation.h"

namespace NN {

double nn_log(double x) {
    return std::log(x + 1e-5);
}

double nn_sigmoid(double x) {
    return 1.0 / (1 + nn_log(x));
}

double nn_tanh(double x) {
    double a = std::exp(x) - std::exp(-1 * x);
    double b = std::exp(x) + std::exp(-1 * x);
    return a * 1.0 / b;
}
    
};
