const navProject = document.querySelector('#nav-project');
const navRun = document.querySelector('#nav-run');
const navStatus = document.querySelector('#nav-status');
const mainConfig = document.querySelector('#main-config');
const mainRecord = document.querySelector('#main-record');
const mainPreview = document.querySelector('#main-preview');
const menuLayer = document.querySelector('#menu-layer');
const panelLayer = document.querySelector('#panel-layer');

const createSeperator = (name, target, before) => {
	const node = create('.seperator', function() {
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
	create('', name, node);
	const figure = create('figure', node);
	create(figure, 2);
	if (before) {
		target.insertBefore(node, target.firstChild);
	}
	else {
		target.appendChild(node);
	}
	return node;
};
const createItem = (name, intro, target, onclick) => {
	const node = create(target, onclick);
	const container = create(node);
	create('', name, container);
	create('', intro, container);
};

const panelLeft = create('#panel-left.panel-list');
const panelRight = create('#panel-right.panel-list');
panelLeft.onclose = () => {
	navProject.classList.remove('active');
};
navProject.addEventListener('click', () => {
	navProject.classList.add('active');
	panelLayer.open(panelLeft);
	panelLeft.innerHTML = '';
	// createSeperator('Local', panelLeft);
	createSeperator(ip === 'localhost' ? 'Local' : 'Remote', panelLeft);

	for (let proj in projectMap) {
		const srcs = projectMap[proj][1].split('\n');
		const recs = projectMap[proj][2].split('\n');
		let nsrc = 0;
		let nrec = 0;
		for (let line of srcs) {
			if (line[0] !== '#' && line.length >= 3) {
				nsrc++;
			}
		}
		for (let line of recs) {
			if (line[0] !== '#' && line.length >= 3) {
				nrec++;
			}
		}
		createItem(proj, `${nsrc} sources, ${nrec} stations`, panelLeft, () => {
			localStorage.setItem('inv_cu_proj', proj);
			window.location.reload();
		});
	}
});
navStatus.addEventListener('click', () => {
	// navStatus.classList.add('active');
	panelLayer.open(panelRight);
});
panelRight.onclose = () => {
	// navStatus.classList.remove('active');
};

let currentProject = null;
const configMap = {};
const configList = [];
const lineMap = {};
const lineList = [];
const projectMap = {};

let ip, password;
try {
	const info = JSON.parse(require('fs').readFileSync(`${__dirname}/../server.json`));
	ip = info.ip;
	password = info.password;
}
catch (e) {
	require('../server.js');
	ip = 'localhost';
	password = '';
}

const ws = new WebSocket(`ws://${ip}:8080`);

const updateConfig = (cfg, write) => {
	if (!cfg.hasOwnProperty('value')) return;
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
			if (lineMap[cfg.id]) {
				delete lineMap[cfg.id];
				const idx = lineList.indexOf(cfg.id);
				if (idx !== -1) {
					lineList.splice(idx, 1);
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
			if (lineMap[cfg.id]) {
				const line = lineMap[cfg.id].split(' ');
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
				lineMap[cfg.id] = line.join(' ');
			}
			else {
				lineMap[cfg.id] = cfg.id + ' = ' + strvalue;
				if (!lineList.includes(cfg.id)) {
					lineList.push(cfg.id);
				}
			}
		}
		const lines = [];
		for (let id of lineList) {
			if (id.indexOf('__') === 0) {
				lines.push(id.slice(2));
			}
			else if (lineMap[id]) {
				lines.push(lineMap[id]);
			}
		}
		ws.sendJSON('updateConfig', currentProject, lines);
	}
};
const parseConfig = (str, asDefault) => {
	const lines = str.split('\n');
	for (let line of lines) {
		const str = line;
		if (line.indexOf('#') != -1) {
			line = line.slice(0, line.indexOf('#'));
		}
		let idAdded = false;
		if (line.indexOf('=') != -1) {
			line = line.replace(/ /g, '').split('=');
			const cfg = configMap[line[0]];
			if (cfg) {
				let value = '';
				switch(cfg.type) {
					case 'int': case 'list': value = parseInt(line[1]); break;
					case 'float': value = parseFloat(line[1]); break;
					case 'bool': value = (parseInt(line[1]) != 0); break;
					case 'hidden': value = line[1]; break;
				}
				if (asDefault) {
					cfg.defaultValue = value;
					if (!cfg.hasOwnProperty('value')) {
						cfg.value = value;
					}
				}
				else {
					cfg.value = value;
					lineMap[line[0]] = str;
					if (!lineList.includes(line[0])) {
						lineList.push(line[0]);
					}
					idAdded = true;
				}
			}
		}
		if (!idAdded && !asDefault) {
			lineList.push('__' + str);
		}
	}
};
const formatNumber = str => {
	const num = parseFloat(str);
	if (num > 99999) {
		return num.toExponential(2);
	}
	else {
		str = num.toString();
		const idx = str.indexOf('.');
		if (idx !== -1) {
			str = str.slice(0, idx + 3);
		}
		else {
			str += '.00';
		}
		return str;
	}
};
const parseSource = str => {
	createSeperator('Sources', mainRecord);
	const lines = str.split('\n');
	for (let line of lines) {
		const data = line.split(' ');
		if (data.length === 7) {
			createItem(`${formatNumber(data[0])}, ${formatNumber(data[1])}`, `Ricker, ${data[3]}Hz`, mainRecord);
		}
	}
};
const parseStation = str => {
	createSeperator('Stations', mainRecord);
	const lines = str.split('\n');
	let nrec = 0;
	for (let line of lines) {
		const data = line.split(' ');
		if (data.length === 2) {
			nrec++;
			createItem(`${formatNumber(data[0])}, ${formatNumber(data[1])}`, 'Station No.' + nrec, mainRecord);
		}
	}
};
const loadProject = name => {
	currentProject = name;
	for (let i in configMap) {
		delete configMap[i].value;
		delete configMap[i].defaultValue;
	}
	for (let i in lineMap) {
		delete lineMap[i];
	}
	lineList.length = 0;
	const info = projectMap[name];
	parseConfig(info[0]);
	if (configMap.inherit.value) {
		parseConfig(projectMap[configMap.inherit.value][0], true);
	}
	navProject.lastChild.lastChild.innerHTML = name;

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
	for (let cfg of configList) {
		if (cfg.type === 'seperator') {
			cfg.node = createSeperator(cfg.name, mainConfig);
		}
		else if (cfg.id && cfg.type !== 'hidden') {
			cfg.node = create(mainConfig, clickConfig[cfg.type]);
			cfg.node.cfg = cfg;
			const text = create(cfg.node);
			create(text);
			create('', cfg.name, text);
			updateConfig(cfg);
		}
	}
	if (configMap.mode) {
		navRun.lastChild.lastChild.innerHTML = configMap.mode.options[configMap.mode.value];
	}

	parseSource(info[1]);
	parseStation(info[2]);

	mainPreview.tm = createSeperator('True Model', mainPreview);
	const url = `http://${ip}:8081/projects/${currentProject}`;

	if (configMap.inv_mu.value) {
		create('img', mainPreview).src = `${url}/model_true/proc000000_vs.png`;
	}
	if (configMap.inv_lambda.value) {
		create('img', mainPreview).src = `${url}/model_true/proc000000_vp.png`;
	}
	createSeperator('Initial Model', mainPreview);
	if (configMap.inv_mu.value) {
		create('img', mainPreview).src = `${url}/model_init/proc000000_vs.png`;
	}
	if (configMap.inv_lambda.value) {
		create('img', mainPreview).src = `${url}/model_init/proc000000_vp.png`;
	}
};

navRun.addEventListener('click', () => {
	if (navRun.classList.contains('active')) {
		if (confirm('Are your sure to stop current task?')) {
			window.location.reload();
		}
	}
	else {
		navRun.classList.add('active');
		navRun.lastChild.firstChild.innerHTML = 'Running';
		while (mainPreview.firstChild !== mainPreview.tm) {
			mainPreview.firstChild.remove();
		}
		if (panelRight.childElementCount) {
			create('', '&nbsp;', panelRight);
		}
		ws.sendJSON('run', currentProject)
	}
});

const commands = {
	projects(cfgs, projs, strs, srcs, recs, modelinfo) {
		for (let cfg of cfgs) {
			configList.push(cfg);
			if (cfg.id) {
				configMap[cfg.id] = cfg;
			}
		}
		for (let i = 0; i < projs.length; i++) {
			projectMap[projs[i]] = [strs[i], srcs[i], recs[i]];
		}
		const saved = localStorage.getItem('inv_cu_proj');
		if (projs.includes(saved)) {
			loadProject(saved);
		}
		else {
			loadProject(projs[0]);
		}
		console.log(modelinfo)
	},
	task(str) {
		navStatus.firstChild.innerHTML = str;
		let div = 'div';
		if (str.indexOf('  ') === 0) {
			div = 'div class="span"';
		}
		create('', `<${div}>${str.replace(/ {2}/g, '&nbsp;&nbsp;&nbsp;&nbsp;<span>').replace(/ /g, '&nbsp;')}</div>`, panelRight);
	},
	plot(num, info_vp, info_vs, info_rho) {
		const frag = document.createDocumentFragment();
		const url = `http://${ip}:8081/projects/${currentProject}/output`;
		if (configMap.inv_mu.value) {
			const img = create('img');
			img.src = `${url}/proc00000${num}_vs.png`;
			frag.appendChild(img);
		}
		if (configMap.inv_lambda.value) {
			const img = create('img');
			img.src = `${url}/proc00000${num}_vp.png`;
			frag.appendChild(img);
		}
		if (configMap.inv_rho.value) {
			const img = create('img');
			img.src = `${url}/proc00000${num}_rho.png`;
			frag.appendChild(img);
		}
		// from here: img chrome; on-demand plot model
		mainPreview.insertBefore(frag, mainPreview.firstChild);
		createSeperator(`Iteration ${num}`, mainPreview, true);
	},
	model(folder, file) {
		console.log(folder, file);
	},
	done() {
		navRun.lastChild.firstChild.innerHTML = 'Run';
		navRun.classList.remove('active');
	}
};

ws.sendJSON = (...args) => {
	ws.send(JSON.stringify(args));
};
ws.onmessage = msg => {
	const args = JSON.parse(msg.data);
	commands[args.shift()].apply(ws, args);
};
ws.onopen = () => {
	ws.sendJSON('password', password);
};
