

namespace nn {

class NN {
public:
    NN();
    void forward();
    //���򴫲�
    void backword();
private:
   void add_layer(uint32_t node_num,
                  uint32_t level,
                  std::string& acti_fun_name);
   //����ڵ�
   std::vector<double> input_nodes;
   //����ڵ����
   uint32_t input_num;

   //��¼���еĲ�
   std::vector<Layer> layers;
};
    
}
