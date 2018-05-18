const project_root = __dirname + '/../../';
const fs = require('fs');
const getdirs = (path, callback) => {
	fs.readdir(path, (err, files) => {
		if (!err) {
			const dirs = [];
			for (let file of files) {
				if (fs.statSync(path + file).isDirectory) {
					dirs.push(file);
				}
			}
			callback(dirs);
		}
	});
};
getdirs(project_root + 'projects/', dirs => {
	// console.log(dirs);
	return dirs;
});

fs.readFile(project_root + 'gui/config.json', 'utf8', (err, data) => {
	const config = JSON.parse(data);
	for (let i of config) {
		console.log(i);
	}
});