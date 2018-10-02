#pragma once

#include "activation.h"

namespace NN {

class Node {
  public:
    //节点激活函数之前的值
    double b_value;

    //节点激活函数之后的值
    double a_value;

    //偏置
    double b;

    //b_value对loss导数
    double devi_b_value;

    //a_value对loss导数
    double devi_a_value;
    
    //激活函数指针
    double (*activation)(double);

    //激活导数函数指针
    double (*activation_devi)(double);

  public:
    Node() {
        b_value = 0.0;
        a_value = 0.0;
        b = 0;
        devi_b_value = 0.0;
    }
};

}
