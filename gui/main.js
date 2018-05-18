const {app, Menu, BrowserWindow} = require('electron');

let win;
const darwin = process.platform === 'darwin';

function createWindow() {
	win = new BrowserWindow({
		width: 960,
		height: 1440,
		title: 'inv.cu',
		titleBarStyle: darwin ? 'hidden' : ''
	});
	win.loadURL(`file:///Users/widget/Documents/workspace/inv.cu/gui/index.html`);
	win.webContents.openDevTools();
	win.on('closed', () => {
		win = null;
	});

	if (darwin) {
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

		win.webContents.on('did-finish-load', () => {
			win.webContents.executeJavaScript('document.body.classList.add("darwin")');
		});
		win.on('enter-full-screen', () => {
			win.webContents.executeJavaScript('document.body.classList.add("full-screen")');
		});
		win.on('leave-full-screen', () => {
			win.webContents.executeJavaScript('document.body.classList.remove("full-screen")');
		});
	}
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
