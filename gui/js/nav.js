const nav = create('nav', document.body);

const project = create('section#nav-project', nav);
const run = create('section#nav-run', nav);
const status = create('section#nav-status', nav);

const project_figure = create('figure', project);
create(project_figure, 6);
const project_text = create(project);
create(project_text, '', 'Current Project');
create(project_text, '', 'none');

const run_figure = create('figure', run);
create(run_figure);
const run_text = create(run);
create(run_text, '', 'Run');
create(run_text, '', 'none');

const status_figure = create('figure', status);
create(status_figure, 2);