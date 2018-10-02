

namespace nn {

class NN {
public:
    NN();
    
    //���򴫲�
    void forward();

    //���򴫲�
    void backword();

    //�������loss�����һ��������ݶ�
    double calc_last_layer_grad(const std::vector<double>& last_layer,
                                const std::vector<double>& labels,
                                const std::string& loss_type);

    //������������ݶ�
    double calc_other_layer_grad();

private:
   void add_layer(uint32_t node_num,
                  uint32_t level,
                  std::string& acti_fun_name);
   //����ڵ�
   std::vector<double> input_nodes;

   //labelֵ
   std::vector<double> labels;

   //��¼���еĲ�
   std::vector<Layer> _layers;
};
    
}
