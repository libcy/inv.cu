#pragma once

class Source {
protected:
	float f0;
	float t0;
	float angle;
	float amp;

public:
	size_t nt;
	float dt;
	float *stf_x;
	float *stf_y;
	float *stf_z;
	virtual void create(float *, float, float) = 0;
	virtual void init(float *cfg, size_t nt0, float dt0) {
		f0 = cfg[0];
		t0 = cfg[1];
		angle = cfg[2];
		amp = cfg[3];
		nt = nt0;
		dt = dt0;

		stf_x = host::create(nt);
		stf_y = host::create(nt);
		stf_z = host::create(nt);

		float *stf = host::create(nt);
		create(stf, f0, t0);
		for (int it = 0; it < nt; it++) {
			stf_x[it] = amp * stf[it] * cos(angle);
			stf_y[it] = amp * stf[it];
			stf_z[it] = amp * stf[it] * sin(angle);
		}
		free(stf);
	};
	virtual ~Source() {
		free(stf_x);
		free(stf_y);
		free(stf_z);
	};
};
