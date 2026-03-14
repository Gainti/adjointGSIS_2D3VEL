#include "utils.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace utils {

// 修剪空格的部分
std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

int parse_int(const std::string& s) {
    return static_cast<int>(std::stol(s, nullptr, 16));
}

double parse_double(const std::string& s) {
    return std::stod(s);
}

std::string dirname(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    if (b.empty()) {
        return a;
    }
    char sep = '/';
    if (a.find('\\') != std::string::npos && a.find('/') == std::string::npos) {
        sep = '\\';
    }
    if (a.back() == '/' || a.back() == '\\') {
        return a + b;
    }
    return a + sep + b;
}

} // namespace utils
