#include "layer.h"

namespace NN {

/**
 * @brief : ��ʼ����֮ǰ��mat, ��ʼ��ֵΪ0
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
