#pragma once

using std::string;
using std::map;
using std::vector;

class Config {
private:
	void parseConfig(map<string, string> &cfg) {
		for (auto const& x : cfg) {
			if (x.first != "config" && x.first != "inherit") {
				std::istringstream f_valuestream(x.second);
				float f_value;
				f_valuestream >> f_value;
				if (f_valuestream.eof() && !f_valuestream.fail()) {
					f[x.first] = f_value;
					i[x.first] = std::round(f_value);
				}
			}
		}
	};
	void loadConfig(string cfgpath, map<string, string> &cfg) {
		std::ifstream infile(cfgpath);
		size_t pos;
		string data;
		while (getline(infile, data)) {
			pos = data.find_first_not_of(" \t");
			if (pos != string::npos) {
				data = data.substr(pos);
			}
			pos = data.find("=");
			if (pos != string::npos) {
				std::istringstream keystream(data.substr(0, pos));
				std::istringstream valuestream(data.substr(pos + 1));
				string key, value;
				keystream >> key;
				valuestream >> value;
				if (key.size() && value.size() && key[0] != '#' && !cfg[key].size()) {
					cfg[key] = value;
				}
			}
		}
		infile.close();
	};
	void loadSource() {
		size_t pos;
		string data;
		std::ifstream infile(path + "/sources.dat");
		while (getline(infile, data)) {
			pos = data.find_first_not_of(" \t");
			if (pos != string::npos) {
				data = data.substr(pos);
			}
			float *source = host::create(7);
			std::istringstream sourcestream(data);
			for (size_t i = 0; i < 7; i++) {
				sourcestream >> source[i];
			}
			if (sourcestream.eof() && !sourcestream.fail()) {
				src.push_back(source);
			}
		}
		infile.close();
	};
	void loadStation() {
		size_t pos;
		string data;
		std::ifstream infile(path + "/stations.dat");
		while (getline(infile, data)) {
			pos = data.find_first_not_of(" \t");
			if (pos != string::npos) {
				data = data.substr(pos);
			}
			float *station = host::create(2);
			std::istringstream stationstream(data);
			for (size_t i = 0; i < 2; i++) {
				stationstream >> station[i];
			}
			if (stationstream.eof() && !stationstream.fail()) {
				rec.push_back(station);
			}
		}
		infile.close();
	};

public:
	vector<float*> src;
	vector<float*> rec;
	map<string, float> f;
	map<string, int> i;
	string path;
	Config(map<string, string> &cfg) {
		path = cfg["config"];
		loadConfig(path + "/config.ini", cfg);
		if (cfg["inherit"].size()) {
			loadConfig(cfg["inherit"], cfg);
		}
		parseConfig(cfg);
		loadSource();
		loadStation();
		if (i["clean"]) {
			removeDirectory(path + "/output");
		}
		createDirectory(path + "/output");
		if (this->i["nthread"]) {
			device::nthread = this->i["nthread"];
		}
	};
};
