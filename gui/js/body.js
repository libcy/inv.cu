let platform = 'default';
switch (navigator.platform) {
	case 'MacIntel': platform = 'mac'; break;
	case 'Win32': case 'Win64': platform = 'win'; break;
}
document.body.dataset.platform = platform;

const {webFrame} = require('electron');
webFrame.setVisualZoomLevelLimits(1, 1);
webFrame.setLayoutZoomLevelLimits(1, 1);
