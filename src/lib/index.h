#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <ctime>
#include <map>
#include <vector>
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "mkdir.h"
#include "query.h"
#include "device.h"
#include "config.h"
#include "../source/source.h"
#include "../source/ricker.h"
#include "../solver/solver.h"
#include "../solver/fdm.h"
#include "../filter/filter.h"
#include "../filter/gaussian.h"
#include "../misfit/misfit.h"
#include "../misfit/waveform.h"
#include "../misfit/envelope.h"
#include "../optimizer/optimizer.h"
#include "../optimizer/nlcg.h"
#include "../optimizer/lbfgs.h"

namespace module {
	Solver *solver(size_t solver_id) {
		switch (solver_id) {
			case 0: return new FdmSolver();
			default: return nullptr;
		}
	}
	Source *source(size_t source_id) {
		switch (source_id) {
			case 0: return new RickerSource();
			default: return nullptr;
		}
	}
	Filter *filter(size_t filter_id) {
		switch (filter_id) {
			case 0: return new GaussianFilter();
			default: return nullptr;
		}
	}
	Misfit *misfit(size_t misfit_id) {
		switch(misfit_id) {
			case 0: return new WaveformMisfit();
			case 1: return new EnvelopeMisfit();
			default: return nullptr;
		}
	}
	Optimizer *optimizer(size_t optimizer_id) {
		switch(optimizer_id) {
			case 0: return new NLCGOptimizer();
			case 1: return new LBFGSOptimizer();
			default: return nullptr;
		}
	}
}
