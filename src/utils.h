#pragma once

#include <string>
#include <vector>
#include <fstream>

namespace NN {
    
void split(const std::string & input,
           std::vector<std::string> & output,
           const std::string & separator);

void read_lines(const std::string& data_path,
                std::vector<std::string>& lines);

}
