#pragma once

class Misfit {
protected:
	Solver *solver;
	Filter *filter;
	float **obs_x;
	float **obs_y;
	float **obs_z;
	float *trace;

	virtual float run(float *, float *, float *) = 0;

public:
	float ref;
	float run(bool kernel = false, bool snapshot = false) {
		float misfit = 0;
		size_t &nsrc = solver->nsrc, &nrec = solver->nrec, &nt = solver->nt;
		Dim dim(nt, nrec);
		if (kernel) {
			solver->initKernels();
		}
		if(!solver->sh){
			device::init(solver->adstf_y, 0.0f, dim);
		}
		if(!solver->psv){
			device::init(solver->adstf_x, 0.0f, dim);
			device::init(solver->adstf_z, 0.0f, dim);
		}
		for (size_t isrc = 0; isrc < nsrc; isrc++) {
			solver->runForward(isrc);
			for (size_t irec = 0; irec < nrec; irec++) {
				size_t irt = irec * nt;
				if (solver->sh) {
					host::toDevice(trace, obs_y[isrc], dim);
					misfit += run(solver->out_y + irt, trace + irt, solver->adstf_y + irt);
				}
				if (solver->psv) {
					host::toDevice(trace, obs_x[isrc], dim);
					misfit += run(solver->out_x + irt, trace + irt, solver->adstf_x + irt);
					host::toDevice(trace, obs_z[isrc], dim);
					misfit += run(solver->out_z + irt, trace + irt, solver->adstf_z + irt);
				}
			}
			if (kernel) {
				solver->runAdjoint(isrc, snapshot);
			}
		}
		if (kernel && filter) {
			filter->apply(solver->k_lambda);
			filter->apply(solver->k_mu);
			filter->apply(solver->k_rho);
		}
		return misfit;
	};
	void importTraces(size_t isrc, string &path) {
		size_t &nrec = solver->nrec, &nt = solver->nt, &trace_type = solver->trace_type;

		int header1[28];
		short int header2[2];
		short int header3[2];
		float header4[30];

		auto filename = [&](string comp) {
			string istr = std::to_string(isrc);
			for (size_t i = istr.size(); i < 6; i++) {
				istr = "0" + istr;
			}
			return path + "/traces/" + (trace_type?"u":"v") + comp + "_" + istr + ".su";
		};
		auto read = [&](string comp, float *data) {
			std::ifstream infile(filename(comp), std::ifstream::binary);
			for (size_t ir = 0; ir < nrec; ir++) {
				infile.read(reinterpret_cast<char*>(header1), 28 * sizeof(int));
				infile.read(reinterpret_cast<char*>(header2), 2 * sizeof(short int));
				infile.read(reinterpret_cast<char*>(header3), 2 * sizeof(short int));
				infile.read(reinterpret_cast<char*>(header4), 30 * sizeof(float));
				infile.read(reinterpret_cast<char*>(data + ir * nt), nt * sizeof(float));
			}
			infile.close();
		};

		if (solver->sh) {
			read("y", obs_y[isrc]);
		}
		if (solver->psv) {
			read("x", obs_x[isrc]);
			read("z", obs_z[isrc]);
		}
	};
	virtual void init(Config *config, Solver *solver, Filter *filter = nullptr) {
		this->solver = solver;
		this->filter = filter;

		solver->init(config);
		filter->init(solver->dim.nx, solver->dim.nz, config->i["filter_sigma"]);

		ref = 1;

		size_t &nsrc = solver->nsrc, &nrec = solver->nrec, &nt = solver->nt;
		Dim dim(nt, nrec);
		obs_x = host::create2D(nsrc, dim);
		obs_y = host::create2D(nsrc, dim);
		obs_z = host::create2D(nsrc, dim);
		trace = device::create(dim);
		
		if (config->i["trace_file"]) {
			for (size_t is = 0; is < nsrc; is++) {
				this->importTraces(is, config->path);
			}
			if (solver->trace_type == 0) {
				float &dt = solver->dt;
				auto v2u = [&](float *data) {
					data[0] *= dt;
					for (size_t it = 1; it < nt; it++) {
						data[it] = data[it - 1] + data[it] * dt;
					}
				};
				for (size_t is = 0; is < nsrc; is++) {
					for (size_t ir = 0; ir < nrec; ir++) {
						if (solver->sh) {
							v2u(obs_y[is] + ir * nt);
						}
						if (solver->psv) {
							v2u(obs_x[is] + ir * nt);
							v2u(obs_z[is] + ir * nt);
						}
					}
				}
			}
		}
		else {
			std::cout << "Generating traces" << std::endl;
			solver->importModel(true);
			for (size_t is = 0; is < nsrc; is++) {
				solver->runForward(is);
				if (solver->sh) {
					device::toHost(obs_y[is], solver->out_y, dim);
				}
				if (solver->psv) {
					device::toHost(obs_x[is], solver->out_x, dim);
					device::toHost(obs_z[is], solver->out_z, dim);
				}
			}
		}

		solver->importModel(false);
	};
	virtual ~Misfit() {};
};
