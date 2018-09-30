#include "layer.h"

namespace NN {

/**
 * @brief : 初始化层之前的mat, 初始化值为0
 **/
void Layer::init(uint32_t m, 
                 uint32_t n,
                 uint32_t level) {
    mat.resize(m);
    for (uint32_t i = 0; i < m; ++i) {
        mat[i].resize(n, 0);
    }

    this->level = level;
}

}
