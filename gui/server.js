const {Server} = require('ws');
const path = require('path');
const fs = require('fs');

const server = new Server({port: 8080});
const cwd = path.dirname(__dirname);

const getdirs = (pathname, callback) => {
	fs.readdir(path.join(cwd, pathname), (err, files) => {
		if (!err) {
			const dirs = [];
			for (let file of files) {
				if (fs.statSync(path.join(pathname, file)).isDirectory) {
					dirs.push(file);
				}
			}
			callback(dirs);
		}
		else {
            console.log(err)
			callback([]);
		}
	});
};
const read = (...args) => fs.readFileSync(path.join(cwd, ...args), 'utf8');

server.on('connection', ws => {
    if (server.clients.size > 1) {
        ws.close();
    }
    else {
        ws.sendJSON = (...args) => {
            ws.send(JSON.stringify(args));
        };
        getdirs('projects', projects => {
            const configs = [];
            for (let name of projects) {
                configs.push(read('projects', name, 'config.ini'));
            }
            ws.sendJSON('projects', JSON.parse(read('gui', 'config.json')), projects, configs);
        })
    }
});
