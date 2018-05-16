#pragma once

namespace _Optimizer {
	__global__ void reduceSystem(const double * __restrict d_in1, double * __restrict d_out1, const double * __restrict d_in2, double * __restrict d_out2, const int M, const int N) {
		int i = blockIdx.x;
		int j = threadIdx.x;

		if ((i < N) && (j < N)){
			d_out1[j * N + i] = d_in1[j * M + i];
			d_out2[j * N + i] = d_in2[j * M + i];
		}
	}
}

class Optimizer {
protected:
	cusolverDnHandle_t solver_handle;
	clock_t time_start;

	Misfit *misfit;
	Solver *solver;

	float **m_new;
	float **m_old;
	float **g_new;
	float **g_old;
	float **p_new;
	float **p_old;

	float *ls_lens;
	float *ls_vals;
	float *ls_gtg;
	float *ls_gtp;

	bool inv_parameter[3];
	size_t inv_cycle;
	size_t inv_iteration;
	float inv_sharpen;

	size_t ls_step;
	float ls_step_max;
	float ls_step_init;
	float ls_thresh;

	size_t eval_count;
	size_t inv_count;
	size_t ls_count;

	float p_amax(float **a) {
		float amax_a = 0;
		for (size_t ip = 0; ip < 3; ip++) {
			if (inv_parameter[ip]) {
				amax_a = std::max(amax_a, device::amax(a[ip], solver->dim));
			}
		}
		return amax_a;
	};
	float p_dot(float **a, float **b) {
		float dot_ab = 0;
		for (size_t ip = 0; ip < 3; ip++) {
			if (inv_parameter[ip]) {
				dot_ab += device::dot(a[ip], b[ip], solver->dim);
			}
		}
		return dot_ab;
	};
	void p_calc(float **c, float ka, float **a) {
		for (size_t ip = 0; ip < 3; ip++) {
			if (inv_parameter[ip]) {
				device::calc(c[ip], ka, a[ip], solver->dim);
			}
		}
	};
	void p_calc(float **c, float ka, float **a, float kb, float **b) {
		for (size_t ip = 0; ip < 3; ip++) {
			if (inv_parameter[ip]) {
				device::calc(c[ip], ka, a[ip], kb, b[ip], solver->dim);
			}
		}
	};
	void p_copy(float **data, float **source) {
		for (size_t ip = 0; ip < 3; ip++) {
			if (inv_parameter[ip]) {
				device::copy(data[ip], source[ip], solver->dim);
			}
		}
	};
	void p_toHost(float **h_data, float **data) {
		for (size_t ip = 0; ip < 3; ip++) {
			if (inv_parameter[ip]) {
				device::toHost(h_data[ip], data[ip], solver->dim);
			}
		}
	};
	void p_toDevice(float **data, float **h_data) {
		for (size_t ip = 0; ip < 3; ip++) {
			if (inv_parameter[ip]) {
				host::toDevice(data[ip], h_data[ip], solver->dim);
			}
		}
	};
	float p_angle(float **p, float **g, float k = 1){
		float xx = p_dot(p, p);
		float yy = p_dot(g, g);
		float xy = k * p_dot(p, g);
		return acos(xy / sqrt(xx * yy));
	};

	void solveQR(double *h_A, double *h_B, double *XC, size_t nrows, size_t ncols){
		using namespace _Optimizer;
		Dim dim(nrows, ncols);
		int work_size = 0;
		int *devInfo = device::create<int>(1);

		double *d_A = device::create<double>(nrows * ncols);
		cudaMemcpy(d_A, h_A, nrows * ncols * sizeof(double), cudaMemcpyHostToDevice);

		double *d_TAU = device::create<double>(min(nrows, ncols));
		cusolverDnDgeqrf_bufferSize(solver_handle, nrows, ncols, d_A, nrows, &work_size);
		double *work = device::create<double>(work_size);

		cusolverDnDgeqrf(solver_handle, nrows, ncols, d_A, nrows, d_TAU, work, work_size, devInfo);

		double *d_Q = device::create<double>(nrows * nrows);
		cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, nrows, ncols, min(nrows, ncols), d_A, nrows, d_TAU, d_Q, nrows, work, work_size, devInfo);

		double *d_C = device::create<double>(nrows * nrows);
		device::init(d_C, 0.0, dim);
		cudaMemcpy(d_C, h_B, nrows * sizeof(double), cudaMemcpyHostToDevice);

		cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrows, ncols, min(nrows, ncols), d_A, nrows, d_TAU, d_C, nrows, work, work_size, devInfo);

		double *d_R = device::create<double>(ncols * ncols);
		double *d_B = device::create<double>(ncols * ncols);
		reduceSystem<<<ncols, ncols>>>(d_A, d_R, d_C, d_B, nrows, ncols);

		const double alpha = 1.;
		cublasDtrsm(device::cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, ncols, ncols,
			&alpha, d_R, ncols, d_B, ncols);
		cudaMemcpy(XC, d_B, ncols * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		cudaFree(d_Q);
		cudaFree(d_R);
		cudaFree(d_TAU);
		cudaFree(devInfo);
		cudaFree(work);
	};
	double polyfit(double *x, double *y, double *p, size_t n){
		double *A = host::create<double>(3 * n);
		for(size_t i = 0; i < n; i++){
			A[i] = x[i] * x[i];
			A[i + n] = x[i];
			A[i + n * 2] = 1;
		}
		solveQR(A, y, p, n, 3);
		double rss = 0;
		for(size_t i = 0; i < n; i++){
			double ei = p[0] * x[i] * x[i] + p[1] * x[i] + p[2];
			rss += pow(y[i] - ei, 2);
		}
		return rss;
	};
	float polyfit(float *fx, float *fy, float *fp, size_t n){
		double *x = host::create<double>(n);
		double *y = host::create<double>(n);
		double *p = host::create<double>(3);
		for(size_t i = 0; i < n; i++){
			x[i] = fx[i];
			y[i] = fy[i];
		}
		float rss = polyfit(x, y, p, n);
		for(size_t i = 0; i < 3; i++){
			fp[i] = p[i];
		}
		free(x);
		free(y);
		free(p);
		return rss;
	};

	size_t getHistory(float* &x, float* &f, size_t step_count) {
		x = host::create(step_count + 1);
		f = host::create(step_count + 1);
		for(size_t i = 0; i < step_count + 1; i++){
			size_t j = ls_count - 1 - step_count + i;
			x[i] = ls_lens[j];
			f[i] = ls_vals[j];
		}
		for(int i = 0; i < step_count + 1; i++){
			for(int j = i + 1; j < step_count + 1; j++){
				if(x[j] < x[i]){
					float tmp;
					tmp = x[i]; x[i] = x[j]; x[j] = tmp;
					tmp = f[i]; f[i] = f[j]; f[j] = tmp;
				}
			}
		}
		size_t update_count = 0;
		for(size_t i = 0; i < ls_count; i++){
			if(fabs(ls_lens[i]) < 1e-6){
				update_count++;
			}
		}
		if (update_count > 0) {
			update_count--;
		}
		return update_count;
	};
	size_t argmin(float *f, int n){
		float min = f[0];
		size_t idx = 0;
		for(size_t i = 1; i < n; i++){
			if(f[i] < min){
				min = f[i];
				idx = i;
			}
		}
		return idx;
	};
	bool checkBracket(float *x, float *f, int n){
		size_t imin = argmin(f, n);
		float fmin = f[imin];
		if(fmin < f[0]){
			for(size_t i = imin; i < n; i++){
				if(f[i] > fmin){
					return true;
				}
			}
		}
		return false;
	};
	bool goodEnough(float *x, float *f, int n, float &x0){
		float thresh = log10(ls_thresh);
		if(!checkBracket(x, f, n)){
			return false;
		}
		float p[3];
		size_t idx = argmin(f, n) - 1;
		size_t fitlen = std::min<size_t>(4, n - idx);
		polyfit(x + idx, f + idx, p, fitlen);
		if(p[0] <= 0){
			std::cout << "  line search error" << std::endl;
		}
		else{
			x0 = -p[1] / (2 * p[0]);
			for(size_t i = 1; i < n; i++){
				if(fabs(log10(x[i] / x0)) < thresh){
					return true;
				}
			}
		}
		return false;
	};
	static float backtrack2(float f0, float g0, float x1, float f1, float b1, float b2){
		float x2 = -g0 * x1 * x1 / (2  *(f1 - f0 - g0 * x1));
		if(x2 > b2 * x1){
			x2 = b2 * x1;
		}
		else if(x2 < b1 * x1){
			x2 = b1 * x1;
		}
		return x2;
	};
	float bracket(size_t step_count, float step_max, int &status) {
		float *x, *f, alpha;
		size_t update_count = getHistory(x, f, step_count);

		if (step_count == 0) {
			if (update_count == 0) {
				alpha = 1 / ls_gtg[inv_count - 1];
				status = 0;
			}
			else {
				size_t idx = argmin(ls_vals, ls_count - 1);
				alpha = ls_lens[idx] * ls_gtp[inv_count - 2] / ls_gtp[inv_count - 1];
				status = 0;
			}
		}
		else if(checkBracket(x, f, step_count + 1)){
			if(goodEnough(x, f, step_count + 1, alpha)){
				alpha = x[argmin(f, step_count + 1)];
				status = 1;
			}
			else{
				status = 0;
			}
		}
		else if(step_count <= ls_step){
			size_t i;
			for(i = 1; i < step_count + 1; i++){
				if(f[i] > f[0]) break;
			}
			if(i == step_count + 1){
				alpha = 1.618034 * x[step_count];
				status = 0;
			}
			else{
				float slope = ls_gtp[inv_count - 1] / ls_gtg[inv_count - 1];
				alpha = backtrack2(f[0], slope, x[1], f[1], 0.1, 0.5);
				status = 0;
			}
		}
		else{
			alpha = 0;
			status = -1;
		}

		if(alpha > step_max){
			if(step_count == 0){
				alpha = 0.618034 * step_max;
				status = 0;
			}
			else{
				alpha = step_max;
				status = 1;
			}
		}

		free(x);
		free(f);

		return alpha;
	};
	float backtrack(size_t step_count, float step_max, int &status) {
		float *x, *f, alpha;
		size_t update_count = getHistory(x, f, step_count);
		size_t idx = argmin(f, step_count + 1);

		if (update_count == 0) {
			alpha = bracket(step_count, step_max, status);
		}
		else if (step_count == 0) {
			alpha = std::min(1.0f, step_max);
			status = 0;
		}
		else if (f[idx] < f[0]) {
			alpha = x[idx];
			status = 1;
		}
		else if (step_count <= ls_step) {
			float slope = ls_gtp[inv_count - 1] / ls_gtg[inv_count - 1];
			alpha = backtrack2(f[0], slope, x[1], f[1], 0.1, 0.5);
			status = 0;
		}
		else {
			alpha = 0;
			status = -1;
		}

		free(x);
		free(f);

		return alpha;
	};

	virtual float calcStep(size_t, float, int &) = 0;
	virtual int lineSearch(float f) {
		int status = 0;
		float alpha = 0;

		float norm_m = p_amax(m_new);
		float norm_p = p_amax(p_new);
		float gtg = p_dot(g_new, g_new);
		float gtp = p_dot(g_new, p_new);

		float step_max = ls_step_max * norm_m / norm_p;
		size_t step_count = 0;
		ls_lens[ls_count] = 0;
		ls_vals[ls_count] = f;
		ls_gtg[inv_count - 1] = gtg;
		ls_gtp[inv_count - 1] = gtp;
		ls_count++;

		float alpha_current = 0;

		if(ls_step_init && ls_count <= 1){
			alpha = ls_step_init * norm_m / norm_p;
		}
		else{
			alpha = calcStep(step_count, step_max, status);
		}

		while(true){
			p_calc(m_new, 1, m_new, alpha - alpha_current, p_new);
			alpha_current = alpha;
			ls_lens[ls_count] = alpha;
			ls_vals[ls_count] = misfit->run(false);
			ls_count++;
			eval_count++;
			step_count++;

			alpha = calcStep(step_count, step_max, status);
			std::cout << "  step " << (step_count < 10 ? "0" : "")
				<< step_count << "  misfit = "
				<< ls_vals[ls_count - 1] / misfit->ref
				<< std::endl;

			if(status > 0){
				std::cout << "  alpha = " << alpha << std::endl;
				p_calc(m_new, 1, m_new, alpha - alpha_current, p_new);
				return status;
			}
			else if(status < 0){
				p_calc(m_new, 1, m_new, -alpha_current, p_new);
				if(p_angle(p_new, g_new, -1) < 1e-3){
					std::cout << "  line search failed" << std::endl;
					return status;
				}
				else{
					std::cout << "  restarting line search..." << std::endl;
					restartSearch();
					return lineSearch(f);
				}
			}
		}
	};
	virtual int computeDirection() = 0;
	virtual void restartSearch() {
		p_calc(p_new, -1, g_new);
		ls_count = 0;
		inv_count = 1;
	};

public:
	void run() {
		solver->exportAxis();
		solver->exportModel();

		time_start = clock();
		for(int iter = 0; iter < inv_iteration; iter++){
			std:: cout << "\nStarting iteration " << iter + 1 << " / " << inv_iteration << std::endl;
			float f = misfit->run(true);
			if (iter == 0) misfit->ref = f;
			eval_count += 2;

			std::cout << "  step 00  misfit = " << f / misfit->ref << std::endl;
			if (computeDirection() < 0) {
				restartSearch();
			}

			p_copy(m_old, m_new);
			p_copy(p_old, p_new);
			p_copy(g_old, g_new);

			if (lineSearch(f) < 0) {
				break;
			}

			solver->exportKernel(iter + 1);
			solver->exportModel(iter + 1);
		}

		float time_elapsed = (float)(clock() - time_start) / CLOCKS_PER_SEC;
		std::cout << "\nFinal misfit: " << misfit->run(false) / misfit->ref << std::endl;
		std::cout << "Elapsed time: ";
		if(time_elapsed > 60){
			size_t t_min = time_elapsed / 60;
			size_t t_sec = time_elapsed - t_min * 60;
			std::cout << t_min << "min " << t_sec << "s" << std::endl;
		}
		else{
			std::cout << std::round(time_elapsed) << std::endl;
		}
	};
	virtual void init(Config *config, Solver *solver, Misfit *misfit) {
		enum Parameter { lambda = 0, mu = 1, rho = 2 };
		this->misfit = misfit;
		this->solver = solver;

		inv_parameter[lambda] = solver->inv_lambda;
		inv_parameter[mu] = solver->inv_mu;
		inv_parameter[rho] = solver->inv_rho;
		inv_iteration = config->i["inv_iteration"];
		inv_cycle = config->i["inv_cycle"];
		inv_sharpen = config->f["inv_sharpen"];

		ls_step = config->i["ls_step"];
		ls_step_max = config->f["ls_step_max"];
		ls_step_init = config->f["ls_step_init"];
		ls_thresh = config->f["ls_thresh"];

		int nstep = inv_iteration * ls_step;
		ls_vals = host::create(nstep);
		ls_lens = host::create(nstep);
		ls_gtg = host::create(inv_iteration);
		ls_gtp = host::create(inv_iteration);

		m_new = host::create<float *>(3);
		g_new = host::create<float *>(3);
		m_old = device::create2D(3, solver->dim);
		g_old = device::create2D(3, solver->dim);
		p_new = device::create2D(3, solver->dim);
		p_old = device::create2D(3, solver->dim);

		m_new[lambda] = solver->lambda;
		m_new[mu] = solver->mu;
		m_new[rho] = solver->rho;
		g_new[lambda] = solver->k_lambda;
		g_new[mu] = solver->k_mu;
		g_new[rho] = solver->k_rho;

		eval_count = 0;
		inv_count = 0;
		ls_count = 0;

		cusolverDnCreate(&solver_handle);
	};
	virtual ~Optimizer() {
		cusolverDnDestroy(solver_handle);
	};
};
