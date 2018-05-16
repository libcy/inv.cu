#pragma once

namespace module {
	Source *source(size_t);
}

class Solver {
protected:
	string path;

	size_t sfe;
	size_t wfe;
	size_t wae;

	size_t abs_width;
	float abs_alpha;
	bool abs_left;
	bool abs_top;
	bool abs_right;
	bool abs_bottom;

	size_t model_npt;
	float *model_x;
	float *model_z;
	float *model_vp;
	float *model_vs;
	float *model_rho;

	float *src_x;
	float *src_z;
	float *rec_x;
	float *rec_z;

	float *stf_x;
	float *stf_y;
	float *stf_z;

	void exportTraces(size_t isrc) {
		int header1[28];
		short int header2[2];
		short int header3[2];
		float header4[30];

		for (size_t i = 0; i < 28; i++) header1[i] = 0;
		for (size_t i = 0; i < 2; i++) header2[i] = 0;
		for (size_t i = 0; i < 2; i++) header3[i] = 0;
		for (size_t i = 0; i < 30; i++) header4[i] = 0;

		float xs = device::get(src_x, isrc);
		float zs = device::get(src_z, isrc);

		float *host_rec_x = host::create(nrec, rec_x);
		float *host_rec_z = host::create(nrec, rec_z);

		short int dt_int2;
		if(dt * 1e6 > pow(2, 15)){
			dt_int2 = 0;
		}
		else{
			dt_int2 = dt * 1e6;
		}

		header1[18] = std::round(xs);
		header1[19] = std::round(zs);

		header2[0] = 0;
		header2[1] = nt;

		header3[0] = dt_int2;
		header3[1] = 0;

		auto filename = [&](string comp) {
			string istr = std::to_string(isrc);
			for (size_t i = istr.size(); i < 6; i++) {
				istr = "0" + istr;
			}
			return path + "/output/" + (trace_type?"u":"v") + comp + "_" + istr + ".su";
		};
		auto write = [&](string comp, float *data) {
			std::ofstream outfile(filename(comp), std::ofstream::binary);
			for (size_t ir = 0; ir < nrec; ir++) {
				header1[0] = ir + 1;
				header1[9] = std::round(host_rec_x[ir] - xs);
				header1[20] = std::round(host_rec_x[ir]);
				header1[21] = std::round(host_rec_z[ir]);
				if(nrec > 1) header4[1] = host_rec_x[1] - host_rec_x[0];

				outfile.write(reinterpret_cast<char*>(header1), 28 * sizeof(int));
				outfile.write(reinterpret_cast<char*>(header2), 2 * sizeof(short int));
				outfile.write(reinterpret_cast<char*>(header3), 2 * sizeof(short int));
				outfile.write(reinterpret_cast<char*>(header4), 30 * sizeof(float));
				outfile.write(reinterpret_cast<char*>(data + ir * nt), nt * sizeof(float));
			}
			outfile.close();
			free(data);
		};

		if (sh) {
			write("y", host::create(nt * nrec, out_y));
		}
		if (psv) {
			write("x", host::create(nt * nrec, out_x));
			write("z", host::create(nt * nrec, out_z));
		}

		free(host_rec_x);
		free(host_rec_z);
	};
	void exportData(string comp, float *data, size_t n = 0) {
		string istr = std::to_string(n);
		for (size_t i = istr.size(); i < 6; i++) {
			istr = "0" + istr;
		}
		int npt = (int) dim;
		std::ofstream outfile(path + "/output/proc" + istr + "_" + comp + ".bin", std::ofstream::binary);
		outfile.write(reinterpret_cast<char*>(&npt), sizeof(int));
		outfile.write(reinterpret_cast<char*>(data), npt * sizeof(float));
		outfile.close();
		free(data);
	};

public:
	bool sh;
	bool psv;
	bool inv_lambda;
	bool inv_mu;
	bool inv_rho;
	bool adj;

	size_t trace_type;
	size_t model_type;

	Dim dim;
	size_t nt;
	size_t nsrc;
	size_t nrec;

	float dx;
	float dz;
	float dt;

	float *adstf_x;
	float *adstf_y;
	float *adstf_z;

	float *out_x;
	float *out_y;
	float *out_z;

	float *lambda;
	float *mu;
	float *rho;

	float *k_lambda;
	float *k_mu;
	float *k_rho;

	virtual void init(Config *config) {
		dt = config->f["dt"];
		nt = config->i["nt"];
		sfe = config->i["sfe"];
		wfe = config->i["wfe"];
		wae = config->i["wae"];
		sh = (bool) config->i["sh"];
		psv = (bool) config->i["psv"];
		adj = (bool) (config->i["mode"] == 0 || config->i["mode"] == 2);

		trace_type = config->i["trace_type"];
		model_type = config->i["model_type"];

		abs_alpha = config->f["abs_alpha"];
		abs_width = config->i["abs_width"];
		abs_left = (bool) config->i["abs_left"];
		abs_top = (bool) config->i["abs_top"];
		abs_right = (bool) config->i["abs_right"];
		abs_bottom = (bool) config->i["abs_bottom"];

		path = config->path;
		nsrc = config->src.size();
		nrec = config->rec.size();

		src_x = device::create(nsrc);
		src_z = device::create(nsrc);
		rec_x = device::create(nrec);
		rec_z = device::create(nrec);

		stf_x = device::create(nsrc * nt);
		stf_y = device::create(nsrc * nt);
		stf_z = device::create(nsrc * nt);

		for (size_t is = 0; is < nsrc; is++) {
			host::toDevice(src_x + is, config->src[is] + 0, 1);
			host::toDevice(src_z + is, config->src[is] + 1, 1);
			Source *src = module::source(config->src[is][2]);
			src->init(config->src[is] + 3, nt, dt);
			host::toDevice(stf_x + is * nt, src->stf_x, nt);
			host::toDevice(stf_y + is * nt, src->stf_y, nt);
			host::toDevice(stf_z + is * nt, src->stf_z, nt);
			delete src;
		}

		for (size_t ir = 0; ir < nrec; ir++) {
			host::toDevice(rec_x + ir, config->rec[ir] + 0, 1);
			host::toDevice(rec_z + ir, config->rec[ir] + 1, 1);
		}

		if (adj) {
			inv_lambda = (bool) config->i["inv_lambda"];
			inv_mu = (bool) config->i["inv_mu"];
			inv_rho = (bool) config->i["inv_rho"];

			adstf_x = device::create(nrec * nt);
			adstf_y = device::create(nrec * nt);
			adstf_z = device::create(nrec * nt);
		}

		if (sh) {
			out_y = device::create(nrec * nt);
		}
		if (psv) {
			out_x = device::create(nrec * nt);
			out_z = device::create(nrec * nt);
		}
	};
	virtual void importModel(bool model) {
		string model_path = model ? "model_true" : "model_init";
		int npt = 0;
		auto read = [&](string comp, float* &data) {
			std::ifstream infile(path + "/" + model_path + "/proc000000_" + comp + ".bin", std::ifstream::binary);
			infile.read(reinterpret_cast<char*>(&npt), sizeof(int));
			if (model_npt != npt) {
				std::cout << "error: invalid model" << std::endl;
			}

			float *buffer = host::create(model_npt);
			infile.read(reinterpret_cast<char*>(buffer), model_npt * sizeof(float));
			data = device::create(model_npt, buffer);
			free(buffer);
			infile.close();
		};
		read("x", model_x);
		read("z", model_z);
		if (model_type) {
			read("lambda", model_vp);
			read("mu", model_vs);
		}
		else {
			read("vp", model_vp);
			read("vs", model_vs);
		}
		read("rho", model_rho);
	};
	virtual void initKernels() = 0;
	virtual void exportAxis() = 0;
	virtual void exportKernel(size_t n = 0) {
		if(inv_lambda) exportData("k_lambda", host::create(dim, k_lambda), n);
		if(inv_mu) exportData("k_mu", host::create(dim, k_mu), n);
		if(inv_rho) exportData("k_rho", host::create(dim, k_rho), n);
	};
	virtual void exportModel(size_t n = 0) {
		if(inv_lambda) exportData("lambda", host::create(dim, lambda), n);
		if(inv_mu) exportData("mu", host::create(dim, mu), n);
		if(inv_rho) exportData("rho", host::create(dim, rho), n);
	};
	virtual void runForward(int, bool = false, bool = false) = 0;
	virtual void runAdjoint(int, bool = false) = 0;
	virtual ~Solver() {};
};
