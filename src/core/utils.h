#pragma once

#include <string>

namespace utils {

std::string trim(const std::string& s);
bool starts_with(const std::string& s, const std::string& p);
int parse_int(const std::string& s);
double parse_double(const std::string& s);
std::string dirname(const std::string& path);
std::string join_path(const std::string& a, const std::string& b);

} // namespace utils
