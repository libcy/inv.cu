#pragma once

size_t getMemoryUsage() {
	size_t free_byte ;
	size_t total_byte ;
	cudaMemGetInfo( &free_byte, &total_byte);
	return free_byte;
}
void checkMemoryUsage(size_t spare_byte){
	size_t free_byte ;
	size_t total_byte ;
	size_t MB = 1024 * 1024;
	cudaMemGetInfo( &free_byte, &total_byte);
	float free_db = (float)free_byte / MB;
	float total_db = (float)spare_byte / MB;
	float used_db = total_db - free_db;

	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(1);
	std::cout << "Memory usage: "
		<< used_db << "MB / " << total_db << "MB" << std::endl;
}
void deviceQuery() {
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess) {
		std::cout << "Failed to get device info: " << cudaGetErrorString(err) << std::endl;
		return;
	}
	if (deviceCount == 0) {
		std::cout << "No CUDA device available" << std::endl;
	}
	else {
		std::cout << deviceCount << " CUDA device(s) detected" << std::endl;
	}

	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(1);
	int driverVersion = 0, runtimeVersion = 0;
	for (size_t idev = 0; idev < deviceCount; idev++) {
		cudaSetDevice(idev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, idev);
		std::cout << "\nDevice " << idev << ": " << deviceProp.name << std::endl;
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		std::cout << "  CUDA Driver Version / Runtime Version          "
			<< driverVersion/1000 << "." <<  (driverVersion%100)/10 << " / "
			<< runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;
		std::cout << "  CUDA Capability Major/Minor version number:    "
			<<  deviceProp.major << "." << deviceProp.minor << std::endl;
		std::cout << "  Total amount of global memory:                 "
			<< (float)deviceProp.totalGlobalMem/1048576.0f << " MB" << std::endl;
		std::cout << "  Maximum number of threads per block:           "
			<< deviceProp.maxThreadsPerBlock << std::endl;
		std::cout << "  Maximum sizes of each dimension of a block:    "
			<< deviceProp.maxThreadsDim[0] << " x "
			<< deviceProp.maxThreadsDim[1] << " x "
			<< deviceProp.maxThreadsDim[2] << std::endl;
		std::cout << "  Maximum sizes of each dimension of a grid:     "
			<< deviceProp.maxGridSize[0] << " x "
			<< deviceProp.maxGridSize[1] << " x "
			<< deviceProp.maxGridSize[2] << std::endl;
	}
}
