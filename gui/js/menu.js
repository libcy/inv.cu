const clickLayer = function() {
	if (this.firstChild) {
		this.firstChild.close();
		this.firstChild.remove();
	}
	this.classList.remove('active');
};
const openLayer = function(node) {
	this.innerHTML = '';
	this.appendChild(node);
	node.addEventListener('click', e => {
		e.stopPropagation();
	});
	this.classList.add('active');
};

create('#panel-layer.layer', document.body, clickLayer).open = openLayer;
create('#menu-layer.layer', document.body, clickLayer).open = openLayer;
