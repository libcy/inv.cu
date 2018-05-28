#pragma once

namespace _LowPassFilter {

}

class LowPassFilter {
protected:
	cufftHandle cufft_handle;
	cufftComplex *fdata;

public:
	void init(size_t nt, size_t nr, float param) {
		using namespace _LowPassFilter;
		Filter::init(nt, nr, param);
		cufftPlan1d(&cufft_handle, nt, CUFFT_C2C, 1);
		cudaMalloc((void**)&fdata, nt * sizeof(cufftComplex));
	};
	void apply(float *data) {
		using namespace _GaussianFilter;
		filterKernelX<<<dim.dg, dim.db>>>(data, gtmp, sigma, dim);
		filterKernelZ<<<dim.dg, dim.db>>>(data, gtmp, gsum, sigma, dim);
	};
	~GaussianFilter() {
		cufftDestroy(cufft_handle);
		cudaFree(fdata);
	};
};
