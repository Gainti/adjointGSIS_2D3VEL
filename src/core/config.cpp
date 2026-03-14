#include "config.h"

#include "utils.h"

#include <fstream>
#include <sstream>

bool Config::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      return false;
    }
    std::string line;
    std::string section = "default";
    while (std::getline(in, line)) {
      auto p = line.find_first_of("#;");
      if (p != std::string::npos) {
        line = line.substr(0, p);
      }
      line = utils::trim(line);
      if (line.empty()) {
        continue;
      }
      if (line.front() == '[' && line.back() == ']') {
        section = utils::trim(line.substr(1, line.size() - 2));
        continue;
      }
      auto eq = line.find('=');
      if (eq == std::string::npos) {
        continue;
      }
      std::string key = utils::trim(line.substr(0, eq));
      std::string val = utils::trim(line.substr(eq + 1));
      data_[section][key] = val;
    }
    return true;
}

std::string Config::get_string(const std::string& section,
                               const std::string& key,
                               const std::string& def) const {
    auto it = data_.find(section);
    if (it == data_.end()) {
      return def;
    }
    auto it2 = it->second.find(key);
    if (it2 == it->second.end()) {
      return def;
    }
    return it2->second;
}

int Config::get_int(const std::string& section,
                    const std::string& key,
                    int def) const {
    auto v = get_string(section, key, "");
    if (v.empty()) {
      return def;
    }
    return std::stoi(v);
}

double Config::get_double(const std::string& section,
                          const std::string& key,
                          double def) const {
    auto v = get_string(section, key, "");
    if (v.empty()) {
      return def;
    }
    return std::stod(v);
}
