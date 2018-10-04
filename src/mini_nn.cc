#include "model.h"

#include <fstream>
#include <string>
#include <map>
#include <algorithm>


void split(const std::string & input,
           std::vector<std::string> & output,
           const std::string & separator) {

    std::string::size_type begin = 0;
    std::string::size_type end = input.find(separator);
    while (std::string::npos != end) {
        output.push_back(input.substr(begin, end - begin));
        begin = end + separator.length();
        end = input.find(separator, begin);
    }
    if (begin != input.length()) {
        output.push_back(input.substr(begin));
    }
}

void read_lines(const std::string& data_path,
                std::vector<std::string>& lines) {

    std::ifstream fin(data_path.c_str());
    std::string line;

    while (getline(fin, line)) {
        lines.push_back(line);
    }
}

//输入格式
//label f1 f2 f3 ... fn, 使用tab分割
void load_data(const std::string& data_path,
               std::vector<std::vector<double> >& x_train,
               std::map<std::string, uint32_t>& all_labels) {
    std::vector<std::string> all_lines;
    {
        read_lines(data_path, all_lines);
        std::random_shuffle(all_lines.begin(), all_lines.end());
    }

    std::vector<double> fea;
    std::vector<std::string> output;
    uint32_t label_num = 0;
    for (uint32_t i = 0; i < all_lines.size(); ++i) {
        fea.clear();
        output.clear();

        split(all_lines[i], output, "\t");
        for (uint32_t j = 0; j < output.size(); ++j) {
            if (j == 0) {
                auto iter = all_labels.find(output[j]);
                if (iter != all_labels.end()) {
                    all_labels[output[j]] = label_num++;
                }
            }else {
                fea.push_back(std::stod(output[j]));
            }
        }
        x_train.push_back(fea);
    }
}

int main() {
    NN::MINI_NN * mini_nn = new NN::MINI_NN();
    mini_nn->add_input_layer(3, 3);
    mini_nn->add_layer(3, "relu");
    //mini_nn->fit();

    return 0;
}
