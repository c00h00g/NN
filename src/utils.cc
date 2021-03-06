#include "utils.h"

namespace NN {

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

    fin.close();
}
    
}
