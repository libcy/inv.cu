#pragma once

namespace _EnvelopeMisfit {
	__global__ void _hilbert(cufftComplex *h, size_t n){
		size_t i = blockIdx.x;
		if(i > 0){
			if(n % 2 == 0){
				if(i < n / 2 + 1){
					h[i].x *= 2;
					h[i].y *= 2;
				}
				else if(i > n / 2 + 1){
					h[i].x = 0;
					h[i].y = 0;
				}
			}
			else{
				if(i < (n + 1) / 2){
					h[i].x *= 2;
					h[i].y *= 2;
				}
				else{
					h[i].x = 0;
					h[i].y = 0;
				}
			}
		}
	}
	__global__ void copyR2C(cufftComplex *a,float *b){
		size_t i = blockIdx.x;
		a[i].x = b[i];
		a[i].y = 0;
	}
	__global__ void copyC2Imag(float *a, cufftComplex *b, int n){
		size_t i = blockIdx.x;
		a[i] = b[i].y / n;
	}
	__global__ void copyC2Abs(float *a, cufftComplex *b, int n){
		size_t i = blockIdx.x;
		a[i] = sqrt(b[i].x*b[i].x + b[i].y*b[i].y) / n;
	}
	__global__ void envelopeTmp(float *etmp, float *esyn, float *eobs, float max){
		size_t it = blockIdx.x;
		etmp[it] = (esyn[it] - eobs[it]) / (esyn[it] + max);
	}
	__global__ void prepareEnvelopeSTF(float *adstf, float *etmp, float *syn, float *ersd, size_t nt){
		size_t it = blockIdx.x;
		adstf[nt - it - 1] = etmp[it] * syn[it] - ersd[it];
	}
}

class EnvelopeMisfit : public Misfit {
protected:
	cufftHandle cufft_handle;
	cufftComplex *csyn;
	cufftComplex *cobs;
	float *esyn;
	float *eobs;
	float *ersd;
	float *etmp;

	float run(float *syn, float *obs, float *adstf){
		using namespace _EnvelopeMisfit;
		size_t &nt = solver->nt;
		Dim dim(nt, 1);
		hilbert(syn, csyn);
		hilbert(obs, cobs);
		copyC2Abs<<<nt, 1>>>(esyn, csyn, nt);
		copyC2Abs<<<nt, 1>>>(eobs, cobs, nt);
		float max = device::amax(esyn, nt) * 0.05;
		if(max < 0.05){
			max = 0.05;
		}
		envelopeTmp<<<nt, 1>>>(etmp, esyn, eobs, max);
		copyC2Imag<<<nt, 1>>>(ersd, csyn, nt);
		device::calc(ersd, ersd, etmp, dim);
		hilbert(ersd, cobs);
		copyC2Imag<<<nt, 1>>>(ersd, cobs, nt);
		prepareEnvelopeSTF<<<nt, 1>>>(adstf, etmp, syn, ersd, nt);
		device::calc(ersd, 1, esyn, -1, eobs, dim);
		return device::norm(ersd, nt) * sqrt(solver->dt);
	};
	void hilbert(float *x, cufftComplex *data){
		using namespace _EnvelopeMisfit;
		size_t &nt = solver->nt;
		copyR2C<<<nt, 1>>>(data, x);
		cufftExecC2C(cufft_handle, data, data, CUFFT_FORWARD);
		_hilbert<<<nt, 1>>>(data, nt);
		cufftExecC2C(cufft_handle, data, data, CUFFT_INVERSE);
	}

public:
	void init(Config *config, Solver *solver, Filter *filter = nullptr) {
		Misfit::init(config, solver, filter);
		size_t &nt = solver->nt;
		cudaMalloc((void**)&csyn, nt * sizeof(cufftComplex));
		cudaMalloc((void**)&cobs, nt * sizeof(cufftComplex));
		esyn = device::create(nt);
		eobs = device::create(nt);
		ersd = device::create(nt);
		etmp = device::create(nt);
		cufftPlan1d(&cufft_handle, nt, CUFFT_C2C, 1);
	};
	~EnvelopeMisfit() {
		cufftDestroy(cufft_handle);
	}
};
