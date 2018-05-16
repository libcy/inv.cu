#pragma once

class LBFGSOptimizer : public Optimizer {
protected:
	size_t lbfgs_mem;
	size_t lbfgs_used;
	float lbfgs_thresh;
	float ***lbfgs_S;
	float ***lbfgs_Y;

	float calcStep(size_t step_count, float step_max, int &status) {
		return backtrack(step_count, step_max, status);
	};
	void restartSearch() {
		Optimizer::restartSearch();
		lbfgs_used = 0;
	};
	int computeDirection() {
		inv_count++;
		if (inv_count == 1) {
			p_calc(p_new, -1, g_new);
			return 0;
		}
		else if(inv_cycle && inv_cycle < inv_count) {
			std::cout << "  restarting LBFGS... [periodic restart]" << std::endl;
			return -1;
		}
		else {
			float** &s0 = lbfgs_S[lbfgs_mem - 1];
			float** &y0 = lbfgs_Y[lbfgs_mem - 1];
			for(int i = lbfgs_mem - 1; i > 0; i--){
				lbfgs_S[i] = lbfgs_S[i - 1];
				lbfgs_Y[i] = lbfgs_Y[i - 1];
			}
			lbfgs_S[0] = s0;
			lbfgs_Y[0] = y0;

			float** &s = m_old;
			float** &y = p_old;

			p_calc(s, 1, m_new, -1, m_old);
			p_toHost(lbfgs_S[0], s);
			p_calc(y, 1, g_new, -1, g_old);
			p_toHost(lbfgs_Y[0], y);

			if(lbfgs_used < lbfgs_mem){
				lbfgs_used++;
			}

			float *rh = host::create(lbfgs_used);
			float *al = host::create(lbfgs_used);

			p_copy(p_new, g_new);
			float sty, yty;
			for(size_t i = 0; i < lbfgs_used; i++){
				p_toDevice(s, lbfgs_S[i]);
				p_toDevice(y, lbfgs_Y[i]);
				rh[i] = 1 / p_dot(y, s);
				al[i] = rh[i] * p_dot(s, p_new);
				p_calc(p_new, 1, p_new, -al[i], y);
				if(i == 0){
					sty = 1 / rh[i];
					yty = p_dot(y, y);
				}
			}
			p_calc(p_new, sty / yty, p_new);

			for(int i = lbfgs_used - 1; i >= 0; i--){
				p_toDevice(s, lbfgs_S[i]);
				p_toDevice(y, lbfgs_Y[i]);
				float be = rh[i] * p_dot(y, p_new);
				p_calc(p_new, 1, p_new, al[i] - be, s);
			}

			free(rh);
			free(al);

			float angle = p_angle(g_new, p_new);
			if (angle >= host::pi / 2) {
				std::cout << "  restarting LBFGS... [not a descent direction]" << std::endl;
				return -1;
			}
			else if (angle > host::pi / 2 - lbfgs_thresh) {
				std::cout << "  restarting LBFGS... [practical safeguard]" << std::endl;
				return -1;
			}
			p_calc(p_new, -1, p_new);

			return 1;
		}
	};

public:
	void init(Config *config, Solver *solver, Misfit *misfit) {
		Optimizer::init(config, solver, misfit);
		lbfgs_mem = config->i["lbfgs_mem"];
		lbfgs_thresh = config->f["lbfgs_thresh"];
		lbfgs_used = 0;

		lbfgs_S = host::create<float **>(lbfgs_mem);
		lbfgs_Y = host::create<float **>(lbfgs_mem);
		for (size_t i = 0; i < lbfgs_mem; i++) {
			lbfgs_S[i] = host::create2D(3, solver->dim);
			lbfgs_Y[i] = host::create2D(3, solver->dim);
		}
	};
};
