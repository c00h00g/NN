
namespace NN {

double std_log(double x) {
    return std::log(x + 1e-5);
}

double sigmoid(double x) {
    return 1.0 / (1 + std_log(x));
}

double tanh(double x) {
    double a = std::exp(x) - std::exp(-1 * x);
    double b = std::exp(x) + std::exp(-1 * x);
    return a * 1.0 / b;
}
    
};
