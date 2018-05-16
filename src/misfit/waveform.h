#pragma once

namespace _WaveformMisfit {
	__global__ void diff(float *syn, float *obs, float *adstf, float *misfit, float *tw, size_t nt) {
		size_t it = blockIdx.x;
		misfit[it] = (syn[it] - obs[it]) * tw[it];
		adstf[nt - it - 1] = misfit[it] * tw[it] *2;
	}
	__global__ void taper(float *tw, float dt, int nt){
		int it = blockIdx.x;

		float t_end = (nt - 1) * dt;
		float taper_width = t_end / 10;
		float t_min = taper_width;
		float t_max = t_end - taper_width;

		float t = it * dt;
		if(t <= t_min){
			tw[it] = 0.5 + 0.5 * cosf(device::pi * (t_min - t) / (taper_width));
		}
		else if(t >= t_max){
			tw[it] = 0.5 + 0.5 * cosf(device::pi * (t_max - t) / (taper_width));
		}
		else{
			tw[it] = 1;
		}
	}
}

class WaveformMisfit : public Misfit {
protected:
	float *misfit;
	float *tw;

	float run(float *syn, float *obs, float *adstf){
		using namespace _WaveformMisfit;
		size_t &nt = solver->nt;
		diff<<<nt, 1>>>(syn, obs, adstf, misfit, tw, nt);
		return device::norm(misfit, nt) * sqrt(solver->dt);
	};

public:
	void init(Config *config, Solver *solver, Filter *filter = nullptr) {
		using namespace _WaveformMisfit;
		Misfit::init(config, solver, filter);
		size_t &nt = solver->nt;
		misfit = device::create(nt);
		tw = device::create(nt);
		taper<<<nt, 1>>>(tw, solver->dt, nt);
	};
};
