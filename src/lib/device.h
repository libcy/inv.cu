#pragma once

namespace device {
	size_t nthread = 0;
}

class Dim {
public:
	int nx;
	int nz;
	int dg;
	int db;
	int size;

	__device__ bool operator()(int &i, int &j, int &k) {
		k = this->idx();
		if (k >= this->size) return true;
		if (nz == db) {
			i = blockIdx.x;
			j = threadIdx.x;
		}
		else {
			j = k % nz;
			i = (k - j) / nz;
		}
		return false;
	};
	__device__ bool operator()(int &k) {
		k = this->idx();
		return k >= this->size;
	};
	__device__ int operator()(int di, int dj) {
		return this->idx() + this->idx(di, dj);
	};
	__device__ int idx() {
		return blockIdx.x * db + threadIdx.x;
	};
	__device__ int idx(int i, int j) {
		return i * nz + j;
	};

	Dim() {};
	Dim(int nx, int nz) {
		this->init(nx, nz);
	};
	void init(int nx, int nz) {
		this->nx = nx;
		this->nz = nz;
		this->size = nx * nz;
		if (device::nthread && nx > 1 && nz > 1) {
			db = device::nthread;
			dg = std::ceil((float)size / db);
		}
		else {
			dg = nx;
			db = nz;
		}
	};
	int hk(int i, int j) {
		return i * nz + j;
	};
	operator int() const {
		return size;
	}
};

namespace host {
	const float pi = 3.1415927;
	template<typename T = float>
	T *create(size_t len) {
		T *data = (T *)malloc(len * sizeof(T));
		return data;
	}
	template<typename T = float>
	T *create(size_t len, T *data) {
		T *h_data = (T *)malloc(len * sizeof(T));
		cudaMemcpy(h_data, data, len * sizeof(T), cudaMemcpyDeviceToHost);
		return h_data;
	}
	template<typename T = float>
	T **create2D(size_t n, size_t len) {
		T **data = (T **)malloc(n * sizeof(T *));
		for (size_t i = 0; i < n; i++) {
			data[i] =  (T *)malloc(len * sizeof(T));
		}
		return data;
	}
	template<typename T = float>
	T *toDevice(T *data, T *h_data, size_t len) {
		cudaMemcpy(data, h_data, len * sizeof(T), cudaMemcpyHostToDevice);
		return h_data;
	}
}

namespace device {
	__constant__ float pi = 3.1415927;
	cublasHandle_t cublas_handle = NULL;

	template<typename T = float>
	T *create(size_t len) {
		T *data;
		cudaMalloc((void **)&data, len * sizeof(T));
		return data;
	}
	template<typename T = float>
	T *create(size_t len, T *h_data) {
		T *data;
		cudaMalloc((void **)&data, len * sizeof(T));
		cudaMemcpy(data, h_data, len * sizeof(T), cudaMemcpyHostToDevice);
		return data;
	}
	template<typename T = float>
	T **create2D(size_t n, size_t len) {
		T **data = (T **)malloc(n * sizeof(T *));
		for (size_t i = 0; i < n; i++) {
			cudaMalloc((void **)&data[i], len * sizeof(T));
		}
		return data;
	}
	template<typename T = float>
	T *toHost(T *h_data, T *data, size_t len) {
		cudaMemcpy(h_data, data, len * sizeof(T), cudaMemcpyDeviceToHost);
		return h_data;
	}

	template<typename T = float>
	__global__ void _copy(T *data, T *source, Dim dim) {
		int k; if (dim(k)) return;
		data[k] = source[k];
	}
	template<typename T = float>
	__global__ void _init(T *data, T value, Dim dim){
		int k; if (dim(k)) return;
		data[k] = value;
	}
	template<typename T = float>
	void copy(T*data, T *source, Dim &dim) {
		_copy<<<dim.dg, dim.db>>>(data, source, dim);
	}
	template<typename T = float>
	T *copy(T *source, Dim &dim) {
		T *data = create<T>(dim);
		_copy<<<dim.dg, dim.db>>>(data, source, dim);
		return data;
	}
	template<typename T = float>
	void init(T *data, T value, Dim &dim){
		_init<<<dim.dg, dim.db>>>(data, value, dim);
	}
	template<typename T = float>
	T get(T *a, size_t ia) {
		T b[1];
		toHost(b, a + ia, 1);
		return b[0];
	}

	__global__ void _calc(float *c, float ka, float *a, Dim dim) {
		int k; if (dim(k)) return;
		c[k] = ka * a[k];
	}
	__global__ void _calc(float *c, float ka, float *a, float kb, float *b, Dim dim) {
		int k; if (dim(k)) return;
		c[k] = ka * a[k] + kb * b[k];
	}
	__global__ void _calc(float *c, float *a, float *b, Dim dim) {
		int k; if (dim(k)) return;
		c[k] = a[k] * b[k];
	}
	void calc(float *c, float ka, float *a, Dim &dim) {
		_calc<<<dim.dg, dim.db>>>(c, ka, a, dim);
	}
	void calc(float *c, float ka, float *a, float kb, float *b, Dim &dim) {
		_calc<<<dim.dg, dim.db>>>(c, ka, a, kb, b, dim);
	}
	void calc(float *c, float *a, float *b, Dim &dim) {
		_calc<<<dim.dg, dim.db>>>(c, a, b, dim);
	}
	float amax(float *a, size_t len){
		int index = 0;
		cublasIsamax_v2(cublas_handle, len, a, 1, &index);
		return fabs(get(a, index - 1));
	}
	float norm(float *a, size_t len){
		float norm_a = 0;
		cublasSnrm2_v2(cublas_handle, len, a, 1, &norm_a);
		return norm_a;
	}
	float dot(float *a, float *b, size_t len){
		float dot_ab = 0;
		cublasSdot_v2(cublas_handle, len, a, 1, b, 1, &dot_ab);
		return dot_ab;
	}
}
