#include "../lib/index.h"

namespace _generateChecker {
	__global__ void generate(
		size_t cx, size_t cz, size_t nx, size_t nz,
		float dx, float dz,
		float vp0, float vs0, float rho0,
		float dvp, float dvs, float drho,
		float *x, float *z,
		float *vp, float *vs, float *rho, Dim dim){
		int i, j, k; if (dim(i, j, k)) return;
		x[k] = i * dx;
		z[k] = j * dz;
		float wx = dx * nx * 2 / 3 / (cx + 1);
		float wz = dz * nz * 2 / 3 / (cz + 1);
		size_t idx = (x[k] - wx) / wx / 1.5;
		size_t idz = (z[k] - wz) / wz / 1.5;

		float rx = x[k] - wx - idx * wx * 1.5;
		float rz = z[k] - wz - idz * wz * 1.5;

		if (rx > 0 && rx < wx && rz > 0 && rz < wz) {
			if (idx % 2 == idz % 2) {
				vp[k] = vp0 + dvp;
				vs[k] = vs0 + dvs;
				rho[k] = rho0 + drho;
			}
			else{
				vp[k] = vp0 - dvp;
				vs[k] = vs0 - dvs;
				rho[k] = rho0 - drho;
			}
		}
		else {
			vp[k] = vp0;
			vs[k] = vs0;
			rho[k] = rho0;
		}
	}
}

using std::map;
using std::string;
using namespace _generateChecker;

void generateChecker(
	size_t cx, size_t cz, size_t nx, size_t nz,
	float dx, float dz,
	float vp0, float vs0, float rho0,
	float dvp, float dvs, float drho, size_t sigma) {
	Dim dim(nx, nz);

	float *x = device::create(dim);
	float *z = device::create(dim);
	float *vp = device::create(dim);
	float *vs = device::create(dim);
	float *rho = device::create(dim);

	generate<<<dim.dg, dim.db>>>(
		cx, cz, nx, nz, dx, dz,
		vp0, vs0, rho0, dvp, dvs, drho,
		x, z, vp, vs, rho, dim
	);

	if (sigma) {
		Filter *filter = module::filter(0);
		filter->init(nx, nz, sigma);
		if (dvp > 0.1) filter->apply(vp);
		if (dvs > 0.1) filter->apply(vs);
		if (drho > 0.1) filter->apply(rho);
		delete filter;
	}

	int npt = nx * nz;
	float *buffer = host::create(npt);
	auto write = [&](string comp, float *data) {
		device::toHost(buffer, data, dim);
		std::ofstream outfile("output/proc000000_" + comp + ".bin", std::ofstream::binary);
		outfile.write(reinterpret_cast<char*>(&npt), sizeof(int));
		outfile.write(reinterpret_cast<char*>(buffer), npt * sizeof(float));
		outfile.close();
	};

	write("x", x);
	write("z", z);
	write("vp", vp);
	write("vs", vs);
	write("rho", rho);

	cudaFree(x);
	cudaFree(z);
	cudaFree(vp);
	cudaFree(vs);
	cudaFree(rho);
	free(buffer);
}

int main(int argc, const char *argv[]){
	map<string, float> dict {
		{"cx", 4},
		{"cz", 4},
		{"nx", 201},
		{"nz", 201},
		{"sigma", 3},
		{"dx", 2400},
		{"dz", 2400},
		{"vp0", 0},
		{"vs0", 3500},
		{"rho0", 2600},
		{"dvp", 0},
		{"dvs", 400},
		{"drho", 0}
	};

	for (size_t i = 0; i < argc; i++) {
		string arg = argv[i];
		size_t pos = arg.find("=");
		if (pos != string::npos && arg[0] == '-') {
			string key = arg.substr(1, pos - 1);
			string value = arg.substr(pos + 1);

			std::istringstream f_valuestream(value);
			float f_value;
			f_valuestream >> f_value;
			if (f_valuestream.eof() && !f_valuestream.fail()) {
				dict[key] = f_value;
			}
		}
	}

	createDirectory("output");
	generateChecker(
		dict["cx"], dict["cz"], dict["nx"], dict["nz"],
		dict["dx"], dict["dz"], dict["vp0"], dict["vs0"], dict["rho0"],
		dict["dvp"], dict["dvs"], dict["drho"], dict["sigma"]
	);

	return 0;
}
