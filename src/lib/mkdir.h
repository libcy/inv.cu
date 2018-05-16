#ifdef _WIN32
	#include <direct.h>
#else
	#include <dirent.h>
	#include <unistd.h>
	#include <sys/stat.h>
	#define _mkdir(dir) mkdir(dir, 0777)
	#define _rmdir(dir) rmdir(dir)
#endif

using std::string;

void _rmdir_unimplemented(char *path) {}
void _rmdir_posix(char *path) {

}
void createDirectory(string path){
	size_t len = path.size();
	for (size_t i = 0; i < len; i++) {
		if (path[i] == '/') {
			_mkdir(path.substr(0, i).c_str());
		}
	}
	_mkdir(path.c_str());
}

void removeDirectory(string dir_name_) {
// copied from http://www.morethantechnical.com/2012/10/02/reading-a-directory-in-c-win32-posix/

#ifndef WIN32
//open a directory the POSIX way

	DIR *dp;
	struct dirent *ep;
	dp = opendir (dir_name_.c_str());

	if (dp != NULL)
	{
		while (ep = readdir (dp)) {
			if (ep->d_name[0] != '.')
				remove((dir_name_ + "/" + ep->d_name).c_str());
				// std::cout << ep->d_name << std::endl;
		}

		(void) closedir (dp);
	}
	else {
		// std::cerr << ("Couldn't open the directory");
		return;
	}

#else
//open a directory the WIN32 way
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA fdata;

	if(dir_name_[dir_name_.size()-1] == '\\' || dir_name_[dir_name_.size()-1] == '/') {
		dir_name_ = dir_name_.substr(0,dir_name_.size()-1);
	}

	hFind = FindFirstFile(string(dir_name_).append("\\*").c_str(), &fdata);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (strcmp(fdata.cFileName, ".") != 0 &&
				strcmp(fdata.cFileName, "..") != 0)
			{
				if (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					continue; // a diretory
				}
				else
				{
					remove((dir_name_ + "/" + fdata.cFileName).c_str());
					// std::cout << fdata.cFileName << std::endl;
				}
			}
		}
		while (FindNextFile(hFind, &fdata) != 0);
	} else {
		// std::cerr << "can't open directory\n";
		return;
	}

	if (GetLastError() != ERROR_NO_MORE_FILES)
	{
		FindClose(hFind);
		// std::cerr << "some other error with opening directory: " << GetLastError() << endl;
		return;
	}

	FindClose(hFind);
	hFind = INVALID_HANDLE_VALUE;
#endif

	_rmdir(dir_name_.c_str());
}
