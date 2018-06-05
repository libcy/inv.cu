require('./js/http');
const {Server} = require('ws');
const path = require('path');
const fs = require('fs');
const spawn = require('child_process').spawn;
const exec = require('child_process').exec;

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
        task.stdout.on('data', data => {
            let str = data.toString();
            if (str[str.length - 1] === '\n') {
                str = str.slice(0, str.length - 1);
            }
            this.sendJSON('task', str);
            const idx = str.indexOf('Starting iteration');
            if (idx !== -1) {
                const num = parseInt(str.slice(idx + 18, str.indexOf('/'))) - 1;
                if (num > 0) {
                    exec(`./utils/plot_model.py projects/${project}/output vp ${num} --save`);
                    exec(`./utils/plot_model.py projects/${project}/output vs ${num} --save`, () => {
                        this.sendJSON('plot', num);
                    });
                }
            }
        });
    }
};

server.on('connection', ws => {
    if (server.clients.size > 1) {
        ws.close();
    }
    else {
        ws.sendJSON = (...args) => {
            ws.send(JSON.stringify(args));
        };
        ws.on('message', msg => {
            const args = JSON.parse(msg);
            commands[args.shift()].apply(ws, args);
        });
        getdirs('projects', projects => {
            const strs = [], srcs = [], recs = [];
            for (let name of projects) {
                strs.push(read('projects', name, 'config.ini'));
                srcs.push(read('projects', name, 'sources.dat'));
                recs.push(read('projects', name, 'stations.dat'));
            }
            ws.sendJSON('projects', JSON.parse(read('gui', 'config.json')), projects, strs, srcs, recs);
        })
    }
});
