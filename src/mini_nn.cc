#include "model.h"

#include <fstream>
#include <string>
#include <map>
#include <algorithm>





/**
 * @brief : 将label转换为向量
 **/
void trans_label_to_vector() {
    
}

int main() {
    NN::MINI_NN * mini_nn = new NN::MINI_NN();
    mini_nn->add_input_layer(3, 3);
    mini_nn->add_layer(3, "relu");
    //mini_nn->fit();

    return 0;
}
