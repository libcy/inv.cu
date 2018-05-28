#pragma once

class Filter {
protected:
	Dim dim;
	float param;

public:
	virtual void init(size_t nx, size_t nz, float param) {
		this->dim.init(nx, nz);
		this->param = param;
	};
	virtual void apply(float *) = 0;
	virtual ~Filter() {};
};
