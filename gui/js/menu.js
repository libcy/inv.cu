const clickLayer = function() {
	if (this.firstChild) {
		if (this.firstChild.onclose) {
			this.firstChild.onclose();
		}
		this.firstChild.remove();
	}
	this.classList.remove('active');
};
const openLayer = function(node, e) {
	this.innerHTML = '';
	node.style.left = e.clientX + 'px';
	node.style.top = e.clientY + 'px';
	node.addEventListener('click', e => {
		e.stopPropagation();
	});
	this.classList.add('active');
	this.appendChild(node);
	const top = this.offsetHeight - 10 - node.offsetHeight;
	if (e.clientY > top) {
		node.style.top = top + 'px';
	}
	const input = node.querySelector('input');
	if (input) {
		input.focus();
	}
};
const openPanel = function(node) {
	this.innerHTML = '';
	const container = create(this);
	node.addEventListener('click', e => {
		e.stopPropagation();
	});
	container.appendChild(node);
	container.onclose = node.onclose;
	this.classList.add('active');
};

const panelLayer = create('#panel-layer.layer', document.body, clickLayer);
const menuLayer = create('#menu-layer.layer', document.body, clickLayer);

panelLayer.open = openPanel;
panelLayer.close = clickLayer;
menuLayer.open = openLayer;
menuLayer.close = clickLayer;
