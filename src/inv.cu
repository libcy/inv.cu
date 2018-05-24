#include "lib/index.h"

using std::string;
using std::map;

int main(int argc, const char *argv[]){
	size_t free_memory = getMemoryUsage();
	cublasCreate(&device::cublas_handle);

	map<string, string> cfg;

	for (size_t i = 1; i < argc; i++) {
		string arg = argv[i];
		size_t pos = arg.find("=");
		if (arg[0] == '-') {
			if (pos != string::npos && arg[0] == '-') {
				string key = arg.substr(1, pos - 1);
				string value = arg.substr(pos + 1);
				cfg[key] = value;
			}
		}
		else {
			cfg["config"] = arg;
		}
	}

	if (!cfg["config"].size()) {
		std::cout << "Running default project: example_checker_sh" << std::endl;
		cfg["config"] = "example_checker_sh";
	}

	Config *config = new Config(cfg);
	switch (config->i["mode"]) {
		case 0: {
			Solver *solver = module::solver(config->i["solver"]);
			Filter *filter = module::filter(config->i["filter"]);
			Misfit *misfit = module::misfit(config->i["misfit"]);
			Optimizer *optimizer = module::optimizer(config->i["optimizer"]);

			misfit->init(config, solver, filter);
			optimizer->init(config, solver, misfit);
			optimizer->run();
			checkMemoryUsage(free_memory);
			break;
		}
		case 1: {
			Solver *solver = module::solver(config->i["solver"]);
			solver->init(config);
			solver->importModel(true);
			solver->exportAxis();
			solver->runForward(-1, true, true);
			checkMemoryUsage(free_memory);
			break;
		}
		case 2: {
			Solver *solver = module::solver(config->i["solver"]);
			Filter *filter = module::filter(config->i["filter"]);
			Misfit *misfit = module::misfit(config->i["misfit"]);

			misfit->init(config, solver, filter);
			misfit->run(true, true);
			solver->exportAxis();
			solver->exportKernel();
			checkMemoryUsage(free_memory);
			break;
		}
		case 3: {
			Solver *solver = module::solver(config->i["solver"]);
			solver->init(config);
			solver->importModel(true);
			for (size_t is = 0; is < solver->nsrc; is++) {
				solver->runForward(is, true);
			}
			break;
		}
		case 4: {
			deviceQuery();
			break;
		}
	}

	cublasDestroy(device::cublas_handle);
	return 0;
}
