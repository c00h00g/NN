#pragma once

#include "activation.h"

namespace NN {

class Node {
  public:
    //�ڵ㼤���֮ǰ��ֵ
    double b_value;

    //�ڵ㼤���֮���ֵ
    double a_value;

    //ƫ��
    double b;

    //b_value��loss����
    double devi_b_value;

    //a_value��loss����
    double devi_a_value;
    
    //�����ָ��
    double (*activation)(double);

    //���������ָ��
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
