const {app, Menu, BrowserWindow} = require('electron');
const path = require('path');
const url = require('url');

let win;

function createWindow() {
	win = new BrowserWindow({
		width: 960,
		height: 640,
		title: 'inv.cu',
		titleBarStyle: process.platform === 'darwin' ? 'hidden' : '',
		frame: process.platform != 'win32',
		darkTheme: true
	});
	win.loadURL(url.format({
		pathname: path.join(__dirname, 'index.html'),
		protocol: 'file:',
		slashes: true
	}));
	win.webContents.openDevTools();
	win.on('closed', () => {
		win = null;
	});

	if (process.platform === 'darwin') {
		if (!Menu.getApplicationMenu()) {
			const template = [{
				label: "inv.cu",
				submenu: [
					{ role: 'about' },
					{ type: 'separator' },
					{ role: 'quit' }
				]},{
				label: 'Edit',
				submenu: [
					{ role: 'undo' },
					{ role: 'redo' },
					{ type: 'separator' },
					{ role: 'cut' },
					{ role: 'copy' },
					{ role: 'paste' },
					{ role: 'delete' },
					{ role: 'selectall' }
				]
			}]
			Menu.setApplicationMenu(Menu.buildFromTemplate(template));
		}
	}
	win.on('enter-full-screen', () => {
		win.webContents.executeJavaScript('document.body.classList.add("full-screen")');
	});
	win.on('leave-full-screen', () => {
		win.webContents.executeJavaScript('document.body.classList.remove("full-screen")');
	});
	win.on('maximize', () => {
		win.webContents.executeJavaScript('document.body.classList.add("maximized")');
	});
	win.on('unmaximize', () => {
		win.webContents.executeJavaScript('document.body.classList.remove("maximized")');
	});
}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
	app.quit();
});

app.on('activate', () => {
	if (win === null) {
		createWindow();
	}
});
