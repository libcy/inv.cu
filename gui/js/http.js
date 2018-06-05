const http = require('http');
const fs = require('fs');
const server = new http.Server();
const path = require('path');
const cwd = path.dirname(path.dirname(__dirname));

server.listen(8081);
server.on('request', function(request, response) {
	const url = require('url').parse(request.url);
	const filename = url.pathname.substring(1);
	const reject = content => {
		response.writeHead(404, {'Content-Type': 'text/plain; charset="UTF-8"'});
		response.write(content);
		response.end();
	};
	if (path.extname(filename) === '.png') {
		fs.readFile(path.join(cwd, filename), (err, content) => {
			if(err) {
				reject(err.message);
			}
			else {
				response.writeHead(200, {'Content-Type': 'application/octet-stream'});
				response.write(content);
				response.end();
			}
		});
	}
	else {
		reject('Extname unsupported');
	}
});
