#pragma once

class Filter {
protected:
	Dim dim;
	size_t sigma;

public:
	virtual void init(size_t nx, size_t nz, size_t sigma) {
		this->dim.init(nx, nz);
		this->sigma = sigma;
	};
	virtual void apply(float *) = 0;
	virtual ~Filter() {};
};
