require('./js/http.js');
const {Server} = require('ws');
const path = require('path');
const fs = require('fs');
const spawn = require('child_process').spawn;
const exec = require('child_process').exec;
const plotjs = require('./js/plot.js');

const server = new Server({port: 8080});
const cwd = path.dirname(__dirname);

const getdirs = (pathname, callback) => {
	fs.readdir(path.join(cwd, pathname), (err, files) => {
		if (!err) {
			const dirs = [];
			for (let file of files) {
				if (fs.statSync(path.join(cwd, pathname, file)).isDirectory) {
					dirs.push(file);
				}
			}
			callback(dirs);
		}
		else {
			callback([]);
		}
	});
};
const read = (...args) => fs.readFileSync(path.join(cwd, ...args), 'utf8');
const commands = {
	updateConfig(project, lines) {
		fs.writeFileSync(path.join(cwd, 'projects', project, 'config.ini'), lines.join('\n'), 'utf8')
	},
	run(project) {
		const task = spawn('./inv.out', [project]);
		this.task = task;
		const base = path.join(cwd, 'projects', project, 'output');
		let num = 0;
		const plot = num => {
			plotjs(base, 'vp', num);
			plotjs(base, 'vs', num, )
			this.sendJSON('plot', num);
		};
		task.stdout.on('data', data => {
			let str = data.toString();
			if (str[str.length - 1] === '\n') {
				str = str.slice(0, str.length - 1);
			}
			this.sendJSON('task', str);
			const idx = str.indexOf('Starting iteration');
			if (idx !== -1) {
				num = parseInt(str.slice(idx + 18, str.indexOf('/'))) - 1;
				if (num > 0) {
					plot(num);
				}
			}
			else if (str.indexOf('Final misfit') !== -1) {
				plot(num + 1);
			}
		});
		task.on('close', () => {
			this.sendJSON('done');
		});
	},
	password(password) {
		try {
			const info = JSON.parse(read('gui/server.json'));
			if (info.password !== password) {
				this.close();
				return;
			}
		}
		catch (e) {
			// pass
		}
		getdirs('projects', projects => {
			const strs = [], srcs = [], recs = [];
			for (let project of projects) {
				strs.push(read('projects', project, 'config.ini'));
				srcs.push(read('projects', project, 'sources.dat'));
				recs.push(read('projects', project, 'stations.dat'));
			}
			this.sendJSON('projects', JSON.parse(read('gui', 'config.json')), projects, strs, srcs, recs);
		});
	}
};

getdirs('projects', projects => {
	const plot = (project, folder, file) => {
		const base = path.join(cwd, 'projects', project, folder);
		const dir = path.join(base, `proc000000_${file}`);
		fs.stat(`${dir}.bin`, err => {
			if (!err) {
				fs.stat(`${dir}.png`, err => {
					if (err) {
						plotjs(base, file);
					}
				});
			}
		});
	};

	for (let project of projects) {
		plot(project, 'model_true', 'vp');
		plot(project, 'model_true', 'vs');
		plot(project, 'model_true', 'rho');
		plot(project, 'model_init', 'vp');
		plot(project, 'model_init', 'vs');
		plot(project, 'model_init', 'rho');
	}
});

server.on('connection', ws => {
	if (server.clients.size > 1) {
		ws.close();
	}
	else {
		ws.sendJSON = (...args) => {
			if (ws.readyState === 1) {
				ws.send(JSON.stringify(args));
			}
		};
		ws.on('message', msg => {
			const args = JSON.parse(msg);
			commands[args.shift()].apply(ws, args);
		});
		ws.on('close', () => {
			if (ws.task) {
				ws.task.kill();
			}
		});
	}
});
