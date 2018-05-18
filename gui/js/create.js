global.create = (...args) => {
	let tagName = 'div';
	let id = null;
	let classList = null;
	let parentNode = null;
	let innerHTML = null;
	let num = 0;
	for (let arg of args) {
		if (typeof arg === 'string') {
			if (classList === null) {
				classList = arg.split('.');
				const tag = classList.shift().split('#');
				if (tag[0]) {
					tagName = tag[0];
				}
				if (tag[1]) {
					id = tag[1];
				}
			}
			else {
				innerHTML = arg;
			}
		}
		else if (arg instanceof HTMLElement) {
			parentNode = arg;
		}
		else if (typeof arg === 'number') {
			num = arg;
		}
	}
	const elements = [];
	for (let i = 0; i < Math.max(1, num); i++) {
		const element = document.createElement(tagName);
		if (id) {
			element.id = id;
		}
		if (classList) {
			for (let className of classList) {
				element.classList.add(className);
			}
		}
		if (innerHTML) {
			element.innerHTML = innerHTML;
		}
		if (parentNode) {
			parentNode.appendChild(element);
		}
		if (num === 0) {
			return element;
		}
		else {
			elements.push(element);
		}
	}
	return elements;
};