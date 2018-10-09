#include "model.h"

#include <fstream>
#include <string>
#include <map>
#include <algorithm>

int main() {
    NN::MINI_NN * mini_nn = new NN::MINI_NN("train_data.tst", 100, 0.01);
    mini_nn->add_input_layer(1024, 1024);
    mini_nn->add_layer(10, "sigmoid");
    mini_nn->add_layer(10, "softmax");
    mini_nn->add_loss_func("cross-entropy");
    mini_nn->fit();

    mini_nn->predict("test_data.tst");

    return 0;
}
