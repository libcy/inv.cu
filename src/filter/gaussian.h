#pragma once

namespace _GaussianFilter {
	__device__ float gaussian(size_t x, size_t sigma){
		float xf = (float)x;
		float sigmaf = (float)sigma;
		return (1 / (sqrtf(2 * device::pi) * sigmaf)) * expf(-xf * xf / (2 * sigmaf * sigmaf));
	}
	__global__ void initialiseGaussian(float *gsum, size_t sigma, Dim dim){
		int i, j, k; if (dim(i, j, k)) return;
		float sumx = 0;
		for (size_t n = 0; n < dim.nx; n++) {
			sumx += gaussian(i - n, sigma);
		}
		float sumz = 0;
		for (size_t n = 0; n < dim.nz; n++) {
			sumz += gaussian(j - n, sigma);
		}
		gsum[k] = sumx * sumz;
	}
	__global__ void filterKernelX(float *data, float *gtmp, size_t sigma, Dim dim){
		int i, j, k; if (dim(i, j, k)) return;
		float sumx = 0;
		for (size_t n = 0; n < dim.nx; n++) {
			sumx += gaussian(i - n, sigma) * data[dim.idx(n, j)];
		}
		gtmp[k] = sumx;
	}
	__global__ void filterKernelZ(float *data, float *gtmp, float *gsum, size_t sigma, Dim dim){
		int i, j, k; if (dim(i, j, k)) return;
		float sumz = 0;
		for (size_t n = 0; n < dim.nz; n++) {
			sumz += gaussian(j - n, sigma) * gtmp[dim.idx(i, n)];
		}
		data[k] = sumz / gsum[k];
	}
}

class GaussianFilter : public Filter {
private:
	float *gsum;
	float *gtmp;

public:
	void init(size_t nx, size_t nz, size_t sigma) {
		using namespace _GaussianFilter;
		Filter::init(nx, nz, sigma);
		gsum = device::create(dim);
		gtmp = device::create(dim);
		initialiseGaussian<<<dim.dg, dim.db>>>(gsum, sigma, dim);
	};
	void apply(float *data) {
		using namespace _GaussianFilter;
		filterKernelX<<<dim.dg, dim.db>>>(data, gtmp, sigma, dim);
		filterKernelZ<<<dim.dg, dim.db>>>(data, gtmp, gsum, sigma, dim);
	};
	~GaussianFilter() {
		cudaFree(gsum);
		cudaFree(gtmp);
	};
};
