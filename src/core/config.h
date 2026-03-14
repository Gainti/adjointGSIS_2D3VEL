#pragma once

#include <string>
#include <unordered_map>

class Config {
public:
    bool load(const std::string& path);

    std::string get_string(const std::string& section,
                          const std::string& key,
                          const std::string& def) const;
    int get_int(const std::string& section,
                const std::string& key,
                int def) const;
    double get_double(const std::string& section,
                      const std::string& key,
                      double def) const;

 private:
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> data_;
};