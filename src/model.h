

namespace nn {

class NN {
public:
    NN();
    
    //正向传播
    void forward();

    //反向传播
    void backword();

    //计算各种loss对最后一层输出的梯度
    double calc_last_layer_grad(const std::vector<double>& last_layer,
                                const std::vector<double>& labels,
                                const std::string& loss_type);

    //计算其他层的梯度
    double calc_other_layer_grad();

private:
   void add_layer(uint32_t node_num,
                  uint32_t level,
                  std::string& acti_fun_name);
   //输入节点
   std::vector<double> input_nodes;

   //label值
   std::vector<double> labels;

   //记录所有的层
   std::vector<Layer> _layers;
};
    
}
