#include "model.h"

#include <fstream>
#include <string>

//�����ʽ
//label f1 f2 f3 ... fn, ʹ��tab�ָ�

void load_data(const std::string& data_path) {
    
}


int main() {
    NN::MINI_NN * mini_nn = new NN::MINI_NN();
    mini_nn->add_input_layer(3, 3);
    mini_nn->add_layer(3, "relu");
    //mini_nn->fit();

    return 0;
}
