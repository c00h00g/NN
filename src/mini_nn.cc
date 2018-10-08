#include "model.h"

#include <fstream>
#include <string>
#include <map>
#include <algorithm>

int main() {
    NN::MINI_NN * mini_nn = new NN::MINI_NN("train_data", 1);
    mini_nn->add_input_layer(3, 3);
    mini_nn->add_layer(3, "relu");
    //mini_nn->add_loss_func("cross-entropy");
    //mini_nn->fit();

    return 0;
}
