const {remote} = require('electron');

const header = create('header', document.body);
create(header);
create(header);
const controls = create('section', header);
const minimize = create('#header-minimize', controls, () => {
	remote.BrowserWindow.getFocusedWindow().minimize(); 
});
const maximize = create('#header-maximize', controls, () => {
	const win = remote.BrowserWindow.getFocusedWindow();
	if (win.isMaximized()) {
		win.unmaximize();
	}
	else {
		win.maximize();
	}
});
const close = create('#header-close', controls, () => {
	remote.BrowserWindow.getFocusedWindow().close(); 
});

create(minimize);
create(maximize, 8);
create(close, 2);

if (remote.BrowserWindow.getFocusedWindow().isMaximized()) {
	maximize.classList.add('maximized');
}