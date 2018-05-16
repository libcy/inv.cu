#pragma once

namespace _FdmSolver {
	__global__ void mesh2grid(
			float *lambda, float *mu, float *rho,
			float *device_x, float *device_z,
			float *device_vp, float *device_vs, float *device_rho, size_t npt,
			float dmax, float dx, float dz, Dim dim) {
		int i, j, k; if (dim(i, j, k)) return;
		float x = i * dx;
		float z = j * dz;
		float dmin = dmax;
		for (size_t l = 0; l < npt; l++) {
			float dx = x - device_x[l];
			float dz = z - device_z[l];
			float d = dx * dx + dz * dz;
			if (d < dmin) {
				dmin = d;
				lambda[k] = device_vp[l];
				mu[k] = device_vs[l];
				rho[k] = device_rho[l];
			}
		}
	}
	__global__ void vps2lm(float *lambda, float *mu, float *rho, Dim dim) {
		int k; if (dim(k)) return;
		float &vp = lambda[k];
		float &vs = mu[k];
		if (vp > vs) {
			lambda[k] = rho[k] * (vp * vp - 2 * vs * vs);
		}
		else {
			lambda[k] = 0;
		}
		mu[k] = rho[k] * vs * vs;
	}
	__global__ void lm2vps(float *vp, float *vs, float *rho, Dim dim) {
		int k; if (dim(k)) return;
		float &lambda = vp[k];
		float &mu = vs[k];
		vp[k] = sqrt((lambda + 2 * mu) / rho[k]);
		vs[k] = sqrt(mu / rho[k]);
	}
	__global__ void calcIdx(size_t *coord_n_id, float *coord_n, float Ln, float n){
		size_t i = blockIdx.x;
		coord_n_id[i] = (int)(coord_n[i] / Ln * (n - 1) + 0.5);
	}
	__global__ void initAbsbound(
		float *absbound, size_t abs_width, float abs_alpha,
		bool abs_left, bool abs_right, bool abs_bottom, bool abs_top, Dim dim) {
		int i, j, k; if (dim(i, j, k)) return;
		absbound[k] = 1;

		if (abs_left) {
			if (i + 1 < abs_width) {
				absbound[k] *= exp(-pow(abs_alpha * (abs_width - i - 1), 2));
			}
		}
		if (abs_right) {
			if (i > dim.nx - abs_width) {
				absbound[k] *= exp(-pow(abs_alpha * (abs_width + i - dim.nx), 2));
			}
		}
		if (abs_bottom) {
			if (j > dim.nz - abs_width) {
				absbound[k] *= exp(-pow(abs_alpha * (abs_width + j - dim.nz), 2));
			}
		}
		if (abs_top) {
			if (j + 1 < abs_width) {
			   absbound[k] *= exp(-pow(abs_alpha * (abs_width - j - 1), 2));
		   }
		}
	}

	__global__ void divSY(float *dsy, float *sxy, float *szy, float dx, float dz, Dim dim){
		int i, j, k; if (dim(i, j, k)) return;
		if (i >= 2 && i < dim.nx - 2) {
			dsy[k] = 9 * (sxy[k] - sxy[dim(-1,0)]) / (8 * dx) - (sxy[dim(1,0)] - sxy[dim(-2,0)]) / (24 * dx);
		}
		else{
			dsy[k] = 0;
		}
		if (j >= 2 && j < dim.nz - 2) {
			dsy[k] += 9 * (szy[k] - szy[dim(0,-1)]) / (8 * dz) - (szy[dim(0,1)] - szy[dim(0,-2)]) / (24 * dz);
		}
	}
	__global__ void divSXZ(float *dsx, float *dsz, float *sxx, float *szz, float *sxz, float dx, float dz, Dim dim){
		int i, j, k; if (dim(i, j, k)) return;
		if (i >= 2 && i < dim.nx - 2) {
			dsx[k] = 9 * (sxx[k] - sxx[dim(-1,0)])/(8 * dx) - (sxx[dim(1,0)] - sxx[dim(-2,0)]) / (24 * dx);
			dsz[k] = 9 * (sxz[k] - sxz[dim(-1,0)])/(8 * dx) - (sxz[dim(1,0)] - sxz[dim(-2,0)]) / (24 * dx);
		}
		else{
			dsx[k] = 0;
			dsz[k] = 0;
		}
		if (j >= 2 && j < dim.nz - 2) {
			dsx[k] += 9 * (sxz[k] - sxz[dim(0,-1)])/(8 * dz) - (sxz[dim(0,1)] - sxz[dim(0,-2)]) / (24 * dz);
			dsz[k] += 9 * (szz[k] - szz[dim(0,-1)])/(8 * dz) - (szz[dim(0,1)] - szz[dim(0,-2)]) / (24 * dz);
		}
	}
	__global__ void divVY(float *dvydx, float *dvydz, float *vy, float dx, float dz, Dim dim){
		int i, j, k; if (dim(i, j, k)) return;
		if (i >= 1 && i < dim.nx - 2) {
			dvydx[k] = 9 * (vy[dim(1,0)] - vy[k]) / (8 * dx) - (vy[dim(2,0)] - vy[dim(-1,0)]) / (24 * dx);
		}
		else{
			dvydx[k] = 0;
		}
		if (j >= 1 && j < dim.nz - 2) {
			dvydz[k] = 9 * (vy[dim(0,1)] - vy[k]) / (8 * dz) - (vy[dim(0,2)] - vy[dim(0,-1)]) / (24 * dz);
		}
		else{
			dvydz[k] = 0;
		}
	}
	__global__ void divVXZ(float *dvxdx, float *dvxdz, float *dvzdx, float *dvzdz, float *vx, float *vz, float dx, float dz, Dim dim){
		int i, j, k; if (dim(i, j, k)) return;
		if (i >= 1 && i < dim.nx - 2) {
			dvxdx[k] = 9 * (vx[dim(1,0)] - vx[k]) / (8 * dx) - (vx[dim(2,0)] - vx[dim(-1,0)]) / (24 * dx);
			dvzdx[k] = 9 * (vz[dim(1,0)] - vz[k]) / (8 * dx) - (vz[dim(2,0)] - vz[dim(-1,0)]) / (24 * dx);
		}
		else{
			dvxdx[k] = 0;
			dvzdx[k] = 0;
		}
		if (j >= 1 && j < dim.nz - 2) {
			dvxdz[k] = 9 * (vx[dim(0,1)] - vx[k]) / (8 * dz) - (vx[dim(0,2)] - vx[dim(0,-1)]) / (24 * dz);
			dvzdz[k] = 9 * (vz[dim(0,1)] - vz[k]) / (8 * dz) - (vz[dim(0,2)] - vz[dim(0,-1)]) / (24 * dz);
		}
		else{
			dvxdz[k] = 0;
			dvzdz[k] = 0;
		}
	}

	__global__ void updateSY(float *sxy, float *szy, float *dvydx, float *dvydz, float *mu, float dt, Dim dim){
		int k; if (dim(k)) return;
		sxy[k] += dt * mu[k] * dvydx[k];
		szy[k] += dt * mu[k] * dvydz[k];
	}
	__global__ void updateSXZ(float *sxx, float *szz, float *sxz, float *dvxdx, float *dvxdz, float *dvzdx, float *dvzdz,
		float *lambda, float *mu, float dt, Dim dim){
		int k; if (dim(k)) return;
		sxx[k] += dt * ((lambda[k] + 2 * mu[k]) * dvxdx[k] + lambda[k] * dvzdz[k]);
		szz[k] += dt * ((lambda[k] + 2 * mu[k]) * dvzdz[k] + lambda[k] * dvxdx[k]);
		sxz[k] += dt * (mu[k] * (dvxdz[k] + dvzdx[k]));
	}
	__global__ void updateVY(float *vy, float *uy, float *dsy, float *rho, float *absbound, float dt, Dim dim){
		int k; if (dim(k)) return;
		vy[k] = absbound[k] * (vy[k] + dt * dsy[k] / rho[k]);
		uy[k] += vy[k] * dt;
	}
	__global__ void updateVXZ(float *vx, float *vz, float *ux, float *uz, float *dsx, float *dsz, float *rho, float *absbound, float dt, Dim dim){
		int k; if (dim(k)) return;
		vx[k] = absbound[k] * (vx[k] + dt * dsx[k] / rho[k]);
		vz[k] = absbound[k] * (vz[k] + dt * dsz[k] / rho[k]);
		ux[k] += vx[k] * dt;
		uz[k] += vz[k] * dt;
	}

	__global__ void addSTF(float *dsx, float *dsy, float *dsz, float *stf_x, float *stf_y, float *stf_z,
		size_t *src_x_id, size_t *src_z_id, int isrc, bool sh, bool psv, size_t it, size_t nt, Dim dim){
		size_t is = blockIdx.x;
		size_t xs = src_x_id[is];
		size_t zs = src_z_id[is];
		size_t ks = is * nt + it;
		size_t km = dim.idx(xs, zs);

		if (isrc < 0 || isrc == is) {
			if (sh) {
				dsy[km] += stf_y[ks];
			}
			if (psv) {
				dsx[km] += stf_x[ks];
				dsz[km] += stf_z[ks];
			}
		}
	}
	__global__ void saveRec(float *out_x, float *out_y, float *out_z, float *wx, float *wy, float *wz,
		size_t *rec_x_id, size_t *rec_z_id, bool sh, bool psv, size_t it, size_t nt, Dim dim){
		size_t ir = blockIdx.x;
		size_t xr = rec_x_id[ir];
		size_t zr = rec_z_id[ir];
		size_t kr = ir * nt + it;
		size_t km = dim.idx(xr, zr);

		if(sh){
			out_y[kr] = wy[km];
		}
		if(psv){
			out_x[kr] = wx[km];
			out_z[kr] = wz[km];
		}
	}

	__global__ void interactionRhoY(float *k_rho, float *vy, float *vy_fw, float ndt, Dim dim){
		int k; if (dim(k)) return;
		k_rho[k] -= vy_fw[k] * vy[k] * ndt;
	}
	__global__ void interactionRhoXZ(float *k_rho, float *vx, float *vx_fw, float *vz, float *vz_fw, float ndt, Dim dim){
		int k; if (dim(k)) return;
		k_rho[k] -= (vx_fw[k] * vx[k] + vz_fw[k] * vz[k]) * ndt;
	}
	__global__ void interactionMuY(float *k_mu, float *dvydx, float *dvydx_fw, float *dvydz, float *dvydz_fw, float ndt, Dim dim){
		int k; if (dim(k)) return;
		k_mu[k] -= (dvydx[k] * dvydx_fw[k] + dvydz[k] * dvydz_fw[k]) * ndt;
	}
	__global__ void interactionMuXZ(float *k_mu, float *dvxdx, float *dvxdx_fw, float *dvxdz, float *dvxdz_fw,
		float *dvzdx, float *dvzdx_fw, float *dvzdz, float *dvzdz_fw, float ndt, Dim dim){
		int k; if (dim(k)) return;
		k_mu[k] -= (2 * dvxdx[k] * dvxdx_fw[k] + 2 * dvzdz[k] * dvzdz_fw[k] +
			(dvxdz[k] + dvzdx[k]) * (dvzdx_fw[k] + dvxdz_fw[k])) * ndt;
	}
	__global__ void interactionLambdaXZ(float *k_lambda, float *dvxdx, float *dvxdx_fw, float *dvzdz, float *dvzdz_fw, float ndt, Dim dim){
		int k; if (dim(k)) return;
		k_lambda[k] -= ((dvxdx[k] + dvzdz[k]) * (dvxdx_fw[k] + dvzdz_fw[k])) * ndt;
	}
}

class FdmSolver : public Solver  {
private:
	size_t nsfe;
	float xmax;
	float zmax;

	size_t *src_x_id;
	size_t *src_z_id;
	size_t *rec_x_id;
	size_t *rec_z_id;

	float *absbound;
	float *vx;
	float *vy;
	float *vz;
	float *ux;
	float *uy;
	float *uz;
	float *sxx;
	float *syy;
	float *szz;
	float *sxy;
	float *sxz;
	float *szy;

	float *dsx;
	float *dsy;
	float *dsz;
	float *dvxdx;
	float *dvxdz;
	float *dvydx;
	float *dvydz;
	float *dvzdx;
	float *dvzdz;

	float *dvydx_fw;
	float *dvydz_fw;

	float *dvxdx_fw;
	float *dvxdz_fw;
	float *dvzdx_fw;
	float *dvzdz_fw;

	float **vx_forward;
	float **vy_forward;
	float **vz_forward;

	float **ux_forward;
	float **uy_forward;
	float **uz_forward;

	void divS(Dim &dim) {
		using namespace _FdmSolver;
		if (sh) {
			divSY<<<dim.dg, dim.db>>>(dsy, sxy, szy, dx, dz, dim);
		}
		if (psv) {
			divSXZ<<<dim.dg, dim.db>>>(dsx, dsz, sxx, szz, sxz, dx, dz, dim);
		}
	};
	void divV(Dim &dim) {
		using namespace _FdmSolver;
		if(sh){
			updateVY<<<dim.dg, dim.db>>>(vy, uy, dsy, rho, absbound, dt, dim);
			divVY<<<dim.dg, dim.db>>>(dvydx, dvydz, vy, dx, dz, dim);
			updateSY<<<dim.dg, dim.db>>>(sxy, szy, dvydx, dvydz, mu, dt, dim);
		}
		if(psv){
			updateVXZ<<<dim.dg, dim.db>>>(vx, vz, ux, uz, dsx, dsz, rho, absbound, dt, dim);
			divVXZ<<<dim.dg, dim.db>>>(dvxdx, dvxdz, dvzdx, dvzdz, vx, vz, dx, dz, dim);
			updateSXZ<<<dim.dg, dim.db>>>(sxx, szz, sxz, dvxdx, dvxdz, dvzdx, dvzdz, lambda, mu, dt, dim);
		}
	};
	void initWavefields() {
		if (sh) {
			device::init(vy, 0.0f, dim);
			device::init(uy, 0.0f, dim);
			device::init(sxy, 0.0f, dim);
			device::init(szy, 0.0f, dim);
		}
		if (psv) {
			device::init(vx, 0.0f, dim);
			device::init(vz, 0.0f, dim);
			device::init(ux, 0.0f, dim);
			device::init(uz, 0.0f, dim);
			device::init(sxx, 0.0f, dim);
			device::init(szz, 0.0f, dim);
			device::init(sxz, 0.0f, dim);
		}
	};
	void exportSnapshot(size_t it, int isrc = -1) {
		string istr = "";
		size_t we = wfe;
		if (isrc >= 0) {
			istr = std::to_string(isrc);
			size_t digit = std::to_string(nsrc).size();
			for (size_t i = istr.size(); i < digit; i++) {
				istr = "0" + istr;
			}
			we = wae;
		}
		if (we && (it + 1) % we == 0) {
			switch (trace_type) {
				case 0: {
					if (sh) {
						exportData("vy" + istr, host::create(dim, vy), it + 1);
					}
					if (psv) {
						exportData("vx" + istr, host::create(dim, vx), it + 1);
						exportData("vz" + istr, host::create(dim, vz), it + 1);
					}
					break;
				}
				case 1: {
					if (sh) {
						exportData("uy" + istr, host::create(dim, uy), it + 1);
					}
					if (psv) {
						exportData("ux" + istr, host::create(dim, ux), it + 1);
						exportData("uz" + istr, host::create(dim, uz), it + 1);
					}
					break;
				}
			}
		}
	};

public:
	void init(Config *config) {
		Solver::init(config);
		using namespace _FdmSolver;

		model_npt = 0;
		int npt = 0;
		auto read = [&](string comp, float &max) {
			std::ifstream infile(path + "/model_true/proc000000_" + comp + ".bin", std::ifstream::binary);
			infile.read(reinterpret_cast<char*>(&npt), sizeof(int));
			if (model_npt) {
				if (model_npt != npt) {
					std::cout << "error: invalid model" << std::endl;
				}
			}
			else {
				model_npt = npt;
			}

			float *buffer = host::create(model_npt);
			infile.read(reinterpret_cast<char*>(buffer), model_npt * sizeof(float));
			max = 0;
			for (size_t i = 0; i < model_npt; i++) {
				if (buffer[i] > max) {
					max = buffer[i];
				}
			}
			free(buffer);
			infile.close();
		};
		read("x", xmax);
		read("z", zmax);

		float a = zmax / xmax;
		float b = a + 1;
		float c = 1 - (int)model_npt;
		size_t nx = std::round((-b + sqrt(b * b - 4 * a * c)) / (2 * a)) + 1;
		size_t nz = std::round(model_npt / nx);
		dim.init(nx, nz);
		dx = xmax / (nx - 1);
		dz = zmax / (nz - 1);

		nsfe = nt / sfe;

		lambda = device::create(dim);
		mu = device::create(dim);
		rho = device::create(dim);
		absbound = device::create(dim);

		if (sh) {
			vy = device::create(dim);
			uy = device::create(dim);
			sxy = device::create(dim);
			szy = device::create(dim);
			dsy = device::create(dim);
			dvydx = device::create(dim);
			dvydz = device::create(dim);
		}

		if (psv) {
			vx = device::create(dim);
			vz = device::create(dim);
			ux = device::create(dim);
			uz = device::create(dim);
			sxx = device::create(dim);
			szz = device::create(dim);
			sxz = device::create(dim);
			dsx = device::create(dim);
			dsz = device::create(dim);
			dvxdx = device::create(dim);
			dvxdz = device::create(dim);
			dvzdx = device::create(dim);
			dvzdz = device::create(dim);
		}

		if(adj){
			if(sh){
				dvydx_fw = device::create(dim);
				dvydz_fw = device::create(dim);

				vy_forward = host::create2D(nsfe, dim);
				uy_forward = host::create2D(nsfe, dim);
			}
			if(psv){
				dvxdx_fw = device::create(dim);
				dvxdz_fw = device::create(dim);
				dvzdx_fw = device::create(dim);
				dvzdz_fw = device::create(dim);

				vx_forward = host::create2D(nsfe, dim);
				ux_forward = host::create2D(nsfe, dim);
				vz_forward = host::create2D(nsfe, dim);
				uz_forward = host::create2D(nsfe, dim);
			}

			k_lambda = device::create(dim);
			k_mu = device::create(dim);
			k_rho = device::create(dim);
		}

		src_x_id = device::create<size_t>(nsrc);
		src_z_id = device::create<size_t>(nsrc);
		rec_x_id = device::create<size_t>(nrec);
		rec_z_id = device::create<size_t>(nrec);

		calcIdx<<<nsrc, 1>>>(src_x_id, src_x, xmax, nx);
		calcIdx<<<nsrc, 1>>>(src_z_id, src_z, zmax, nz);
		calcIdx<<<nrec, 1>>>(rec_x_id, rec_x, xmax, nx);
		calcIdx<<<nrec, 1>>>(rec_z_id, rec_z, zmax, nz);

		initAbsbound<<<dim.dg, dim.db>>>(
			absbound, abs_width, abs_alpha, abs_left, abs_right, abs_bottom, abs_top, dim
		);
	};
	void importModel(bool model) {
		Solver::importModel(model);
		using namespace _FdmSolver;

		mesh2grid<<<dim.dg, dim.db>>>(
			lambda, mu, rho,
			model_x, model_z, model_vp, model_vs, model_rho, model_npt,
			xmax * xmax + zmax * zmax, dx, dz, dim
		);

		if (model_type == 0) {
			vps2lm<<<dim.dg, dim.db>>>(lambda, mu, rho, dim);
		}
	};
	void exportModel(size_t n = 0) {
		if (model_type == 0) {
			using namespace _FdmSolver;
			float *lambda1 = device::copy(lambda, dim);
			float *mu1 = device::copy(mu, dim);
			float *rho1 = device::copy(rho, dim);
			lm2vps<<<dim.dg, dim.db>>>(lambda1, mu1, rho1, dim);
			if (inv_lambda && inv_mu) exportData("vp", host::create(dim, lambda1), n);
			if (inv_mu) exportData("vs", host::create(dim, mu1), n);
			if (inv_rho) exportData("rho", host::create(dim, rho1), n);
			cudaFree(lambda1);
			cudaFree(mu1);
			cudaFree(rho1);
		}
		else {
			Solver::exportModel(n);
		}
	};
	void initKernels() {
		device::init(k_lambda, 0.0f, dim);
		device::init(k_mu, 0.0f, dim);
		device::init(k_rho, 0.0f, dim);
	};
	void exportAxis() {
		float *x = host::create(dim);
		float *z = host::create(dim);

		for (size_t i = 0; i < dim.nx; i++) {
			for (size_t j = 0; j < dim.nz; j++) {
				int k = dim.hk(i, j);
				x[k] = i * dx;
				z[k] = j * dz;
			}
		}

		exportData("x", x);
		exportData("z", z);
	};
	void runForward(int isrc, bool trace = false, bool snapshot = false) {
		using namespace _FdmSolver;

		initWavefields();

		for (size_t it = 0; it < nt; it++) {
			int isfe = -1;
			if (sfe && adj && (it + 1) % sfe == 0) {
				isfe = nsfe - (it + 1) / sfe;
			}
			if (isfe >= 0) {
				if(sh){
					device::toHost(uy_forward[isfe], uy, dim);
				}
				if(psv){
					device::toHost(ux_forward[isfe], ux, dim);
					device::toHost(uz_forward[isfe], uz, dim);
				}
			}
			divS(dim);
			addSTF<<<nsrc, 1>>>(
				dsx, dsy, dsz, stf_x, stf_y, stf_z,
				src_x_id, src_z_id, isrc, sh, psv, it, nt, dim
			);
			divV(dim);
			if (adj || trace_type == 1) {
				saveRec<<<nrec, 1>>>(
					out_x, out_y, out_z, ux, uy, uz,
					rec_x_id, rec_z_id, sh, psv, it, nt, dim
				);
			}
			else if(trace_type == 0) {
				saveRec<<<nrec, 1>>>(
					out_x, out_y, out_z, vx, vy, vz,
					rec_x_id, rec_z_id, sh, psv, it, nt, dim
				);
			}
			if (isfe >= 0) {
				if(sh){
					device::toHost(vy_forward[isfe], vy, dim);
				}
				if(psv){
					device::toHost(vx_forward[isfe], vx, dim);
					device::toHost(vz_forward[isfe], vz, dim);
				}
			}

			if (snapshot) {
				exportSnapshot(it);
			}
		}

		if (trace) {
			exportTraces(std::max<int>(isrc, 0));
		}
	};
	void runAdjoint(int isrc, bool snapshot = false) {
		using namespace _FdmSolver;

		initWavefields();

		for (size_t it = 0; it < nt; it++) {
			divS(dim);
			addSTF<<<nrec, 1>>>(
				dsx, dsy, dsz, adstf_x, adstf_y, adstf_z,
				rec_x_id, rec_z_id, -1, sh, psv, it, nt, dim
			);
			divV(dim);
			if ((it + sfe) % sfe == 0) {
				size_t isfe = (it + sfe) / sfe - 1;
				float ndt = sfe * dt;
				if (sh) {
					host::toDevice(dsy, uy_forward[isfe], dim);
					divVY<<<dim.dg, dim.db>>>(dvydx, dvydz, uy, dx, dz, dim);
					divVY<<<dim.dg, dim.db>>>(dvydx_fw, dvydz_fw, dsy, dx, dz, dim);
					host::toDevice(dsy, vy_forward[isfe], dim);
					if (inv_rho) interactionRhoY<<<dim.dg, dim.db>>>(k_rho, vy, dsy, ndt, dim);
					if (inv_mu) interactionMuY<<<dim.dg, dim.db>>>(k_mu, dvydx, dvydx_fw, dvydz, dvydz_fw, ndt, dim);
				}
				if (psv) {
					host::toDevice(dsx, ux_forward[isfe], dim);
					host::toDevice(dsz, uz_forward[isfe], dim);
					divVXZ<<<dim.dg, dim.db>>>(dvxdx, dvxdz, dvzdx, dvzdz, ux, uz, dx, dz, dim);
					divVXZ<<<dim.dg, dim.db>>>(dvxdx_fw, dvxdz_fw, dvzdx_fw, dvzdz_fw, dsx, dsz, dx, dz, dim);

					host::toDevice(dsx, vx_forward[isfe], dim);
					host::toDevice(dsz, vz_forward[isfe], dim);
					if (inv_rho) interactionRhoXZ<<<dim.dg, dim.db>>>(k_rho, vx, dsx, vz, dsz, ndt, dim);
					if (inv_mu) interactionMuXZ<<<dim.dg, dim.db>>>(k_mu, dvxdx, dvxdx_fw, dvxdz, dvxdz_fw, dvzdx, dvzdx_fw, dvzdz, dvzdz_fw, ndt, dim);
					if (inv_lambda) interactionLambdaXZ<<<dim.dg, dim.db>>>(k_lambda, dvxdx, dvxdx_fw, dvzdz, dvzdz_fw, ndt, dim);
				}
			}

			if (snapshot) {
				exportSnapshot(it, isrc);
			}
		}
	};
};
