const path = require('path');
const fs = require('fs');
const cwd = path.dirname(path.dirname(__dirname));
const config = {};

const navProject = document.querySelector('#nav-project');
const navRun = document.querySelector('#nav-run');
const mainConfig = document.querySelector('#main-config');
const menuLayer = document.querySelector('#menu-layer');
const panelLayer = document.querySelector('#panel-layer');

const getdirs = (pathname, callback) => {
	fs.readdir(pathname, (err, files) => {
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
const readConfig = (pathname, callback, asDefault) => {
	fs.readFile(pathname, 'utf8', (err, data) => {
		const lines = data.split('\n');
		for (let line of lines) {
			const str = line;
			if (line.indexOf('#') != -1) {
				line = line.slice(0, line.indexOf('#'));
			}
			let idAdded = false;
			if (line.indexOf('=') != -1) {
				line = line.replace(/ /g, '').split('=');
				const cfg = config[line[0]];
				if (cfg) {
					let value = '';
					switch(cfg.type) {
						case 'int': case 'list': value = parseInt(line[1]); break;
						case 'float': value = parseFloat(line[1]); break;
						case 'bool': value = (parseInt(line[1]) != 0); break;
					}
					if (asDefault) {
						cfg.defaultValue = value;
						if (!cfg.hasOwnProperty(value)) {
							cfg.value = value;
						}
					}
					else {
						cfg.value = value;
						config.lines[line[0]] = str;
						if (!config.lineIds.includes(line[0])) {
							config.lineIds.push(line[0]);
						}
						idAdded = true;
					}
				}
			}
			if (!idAdded) {
				config.lineIds.push('__' + str);
			}
		}
		callback();
	});
};
const updateConfig = (cfg, write) => {
	let str;
	if (cfg.options) {
		str = cfg.options[cfg.value];
	}
	else if (typeof cfg.value === 'number' && cfg.value > 99999) {
		str = cfg.value.toExponential();
	}
	else {
		str = cfg.value.toString();
	}
	const node = cfg.node.firstChild.firstChild;
	node.innerHTML =  cfg.id + ' = ';
	create('span', str, node);

	if (cfg.id === 'mode') {
		navRun.lastChild.lastChild.innerHTML = cfg.options[cfg.value];
	}

	if (write) {
		if (cfg.hasOwnProperty('defaultValue') && cfg.defaultValue === cfg.value) {
			if (config.lines[cfg.id]) {
				delete config.lines[cfg.id];
				let idx = config.lineIds.indexOf(cfg.id);
				if (idx !== -1) {
					config.lineIds.splice(idx, 1);
				}
			}
		}
		else {
			let strvalue;
			if (cfg.type === 'bool') {
				strvalue = cfg.value ? '1' : '0';
			}
			else if (typeof cfg.value === 'number' && cfg.value > 99999) {
				strvalue = cfg.value.toExponential();
			}
			else {
				strvalue = cfg.value.toString();
			}
			if (config.lines[cfg.id]) {
				const line = config.lines[cfg.id].split(' ');
				let eq = false;
				for (let i = 0; i < line.length; i++) {
					if (line[i]) {
						if (eq) {
							let len1 = line[i].length;
							let len2 = strvalue.length;
							if (len1 > len2) {
								for (let j = 0; j < len1 - len2; j++) {
									line.splice(i + 1, 0, '');
								}
							}
							else if(len1 < len2) {
								for (let j = 0; j < len2 - len1; j++) {
									if (line[i + 1] === '') {
										line.splice(i + 1, 1);
									}
								}
							}
							line[i] = strvalue;
							break;
						}
						else if (line[i] === '=') {
							eq = true;
						}
					}
				}
				config.lines[cfg.id] = line.join(' ');
			}
			else {
				config.lines[cfg.id] = cfg.id + ' = ' + strvalue;
				if (!config.lineIds.includes(cfg.id)) {
					config.lineIds.push(cfg.id);
				}
			}
		}
		const lines = [];
		for (let id of config.lineIds) {
			if (id.indexOf('__') === 0) {
				lines.push(id.slice(2));
			}
			else if (config.lines[id]) {
				lines.push(config.lines[id]);
			}
		}
		fs.writeFile(config.path, lines.join('\n'), 'utf8', err => {
			if (err) {
				alert('Failed to write config file.');
			}
		});
	}
};
const createEntries = () => {
	const createInput = cfg => {
		const node = create('.context-menu.prompt');
		const input = create('input', node);
		const buttons = create(node);

		const clickOk = () => {
			let value;
			if (cfg.type === 'int') {
				value = parseInt(input.value);
			}
			else {
				value = parseFloat(input.value);
			}
			if (!isNaN(value)) {
				cfg.value = value;
				updateConfig(cfg, true);
			}
			menuLayer.close();
		};
		const clickCancel = () => {
			menuLayer.close();
		};

		input.value = cfg.value;
		input.type = 'text';
		input.addEventListener('focus', () => {
			input.setSelectionRange(0, input.value.length)
		});
		input.addEventListener('keydown', e => {
			switch (e.keyCode) {
				case 13: clickOk(); break;
				case 27: clickCancel(); break;
			}
		})
		create('', 'ok', buttons, clickOk);
		create('', 'cancel', buttons, clickCancel);
		return node;
	};
	const clickConfig = {
		list(e) {
			const node = create('.context-menu.text');
			for (let option of this.cfg.options) {
				create('', '<span>' + option + '</span>', () => {
					this.cfg.value = this.cfg.options.indexOf(option);
					updateConfig(this.cfg, true);
					menuLayer.close();
				}, node);
			}
			menuLayer.open(node, e);
		},
		int(e) {
			menuLayer.open(createInput(this.cfg), e);
		},
		float(e) {
			menuLayer.open(createInput(this.cfg), e);
		},
		bool() {
			this.cfg.value = !this.cfg.value;
			updateConfig(this.cfg, true);
		}
	};
	for (let cfg of config.entries) {
		if (cfg.type === 'seperator') {
			cfg.node = create('.seperator', mainConfig, function() {
				this.classList.toggle('collapsed');
				let started = false;
				for (let i = 0; i < this.parentNode.childElementCount; i++) {
					const current = this.parentNode.childNodes[i];
					if (!started) {
						if (current === this) {
							started = true;
						}
					}
					else if (current.classList.contains('seperator')) {
						break;
					}
					else {
						if (this.classList.contains('collapsed')) {
							current.style.display = 'none';
						}
						else {
							current.style.display = '';
						}
					}
				}
			});
			create('', cfg.name, cfg.node);
			const figure = create('figure', cfg.node);
			create(figure, 2);
		}
		else if (cfg.id) {
			cfg.node = create(mainConfig, clickConfig[cfg.type]);
			cfg.node.cfg = cfg;
			const text = create(cfg.node);
			create(text);
			create('', cfg.name, text);
			updateConfig(cfg);
		}
	}
	if (config.mode) {
		navRun.lastChild.lastChild.innerHTML = config.mode.options[config.mode.value];
	}
};

const loadProject = global.loadProject = name => {
	config.lines = {};
	config.lineIds = [];
	localStorage.setItem('inv.cu_project', name);
	navProject.lastChild.lastChild.innerHTML = name;
	const project_dir = path.join(cwd, 'projects', name);
	config.path = path.join(project_dir, 'config.ini');
	readConfig(config.path, createEntries);
};

fs.readFile(path.join(cwd, 'gui/config.json'), 'utf8', (err, data) => {
	data = JSON.parse(data);
	config.entries = data;
	for (let entry of data) {
		if (entry.id) {
			config[entry.id] = entry;
		}
	}
	getdirs(path.join(cwd, 'projects'), dirs => {
		const current = localStorage.getItem('inv.cu_project');
		if (current && dirs.includes(current)) {
			loadProject(current);
		}
		else {
			loadProject(dirs[0]);
		}
	});
});

const panelLeft = create('#panel-left.panel-list');
panelLeft.onclose = () => {
	navProject.classList.remove('active');
};
navProject.addEventListener('click', () => {
	navProject.classList.add('active');
	panelLayer.open(panelLeft);
	panelLeft.innerHTML = '';
	getdirs(path.join(cwd, 'projects'), dirs => {
		if (dirs.length) {
			const seperator = create('.seperator', panelLeft);
			create('', 'Local', seperator);
			// from here
		}
	});
});
