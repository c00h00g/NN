#include "model.h"

#include <fstream>
#include <string>
#include <map>
#include <algorithm>

int main() {
    NN::MINI_NN * mini_nn = new NN::MINI_NN("../data/train_data.tst", 30, 0.01);
    mini_nn->add_input_layer(784);
    mini_nn->add_layer(1000, "sigmoid");
    mini_nn->add_layer(800, "sigmoid");
    mini_nn->add_layer(10, "sigmoid");
    mini_nn->add_layer(10, "softmax");
    mini_nn->add_loss_func("cross-entropy");
    mini_nn->fit();

    mini_nn->predict("../data/test_data.tst");

    delete mini_nn;
    mini_nn = NULL;

    return 0;
}
