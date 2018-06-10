'use strict';
const fs = require('fs');
const path = require('path');
const PNG = require('pngjs').PNG;

const info = dir => {
    const buffer = fs.readFileSync(path.join(dir, 'proc000000_z.bin'));
    const npt = buffer.readInt32LE(0);
    const z = [];
    for (let i = 1; i <= npt; i++) {
        let zi = buffer.readFloatLE(i * 4);
        if (z.indexOf(zi) === -1) {
            z.push(zi);
        }
    }
    const nz = z.length;
    const nx = npt / nz;
    return [npt, nx, nz];
};
const name = (comp, num=0) => {
    let str = num.toString();
    while(str.length < 6) {
        str = '0' + str;
    }
    return `proc${str}_${comp}`;
};
const read = (dir, filename) => {
    const buffer = fs.readFileSync(path.join(dir, filename));
    const npt = buffer.readInt32LE(0);
    const data = [];
    for (let i = 1; i <= npt; i++) {
        data.push(buffer.readFloatLE(i * 4));
    }
    return data;
};

module.exports = (dir, comp, num=0) => {
    const [npt, nx, nz] = info(dir);
    const filename = name(comp, num);
    const x = read(dir, 'proc000000_x.bin');
    const z = read(dir, 'proc000000_z.bin');
    const v = read(dir, `${filename}.bin`);
    const dx = Math.max(...x) / (nx - 1);
    const dz = Math.max(...z) / (nz - 1);
    const vmax = Math.max(...v);
    const vmin = Math.min(...v);
    const dv = Math.max(0.01, vmax - vmin);
    const nm = Math.round(nx / 15);
    const ns = Math.round(nx / 15);
    const nx2 = nx + nm + ns;
    const nz2 = nz;

    const png = new PNG({width: nx2, height: nz2});
    const p = [
        [251, 229, 63],
        [96, 198, 104],
        [41, 145, 139],
        [60, 82, 137],
        [67, 6, 82]
    ];
    const np = p.length - 1;
    const dp = 1 / np;
    const set = (ratio, idx) => {
        const ip = Math.min(np - 1, Math.floor(ratio * np));
        const ratio2 = (ratio - dp * ip) * np;
        png.data[idx] = p[ip][0] + (p[ip+1][0] - p[ip][0]) * ratio2;
        png.data[idx + 1] = p[ip][1] + (p[ip+1][1] - p[ip][1]) * ratio2;
        png.data[idx + 2] = p[ip][2] + (p[ip+1][2] - p[ip][2]) * ratio2;
        png.data[idx + 3] = 255;
    };
    for (let i = 0; i < npt; i++) {
        const ix = x[i] / dx;
        const iz = z[i] / dz;
        const idx = (nx2 * iz + ix) << 2;
        let ratio;
        if ((vmax - vmin) / vmax < 1e-6) {
            ratio = 0.5;
        }
        else {
            ratio = (v[i] - vmin) / (vmax - vmin);
        }
        set(ratio, idx);
    }
    for (let ix = 0; ix < ns; ix++) {
        for (let iz = 0; iz < nz; iz++) {
            const idx = nx2 * iz + ix + nx + nm << 2;
            const ratio = iz / (nz - 1);
            set(ratio, idx);
        }
    }
    png.pack().pipe(fs.createWriteStream(path.join(dir, `${filename}.png`)));

};
