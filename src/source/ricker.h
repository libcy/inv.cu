#pragma once

class RickerSource : public Source {
public:
	void create(float *stf, float f0, float t0){
		float amax = 0;
		for (size_t it = 0; it < nt; it++) {
			float t = it * dt - t0;
			float pft = host::pi * f0 * t;
			stf[it] = -t * exp(-pft * pft);

			if(fabs(stf[it]) > amax){
				amax = fabs(stf[it]);
			}
		}
		if(amax > 0){
			for (size_t it = 0; it < nt; it++) {
				stf[it] /= amax;
			}
		}
	}
};
