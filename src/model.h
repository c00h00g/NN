

namespace nn {

class NN {
public:
    NN();
    void forward();
    //反向传播
    void backword();
private:
   void add_layer(uint32_t node_num,
                  uint32_t level,
                  std::string& acti_fun_name);
   //输入节点
   std::vector<double> input_nodes;
   //输入节点个数
   uint32_t input_num;

   //记录所有的层
   std::vector<Layer> layers;
};
    
}
