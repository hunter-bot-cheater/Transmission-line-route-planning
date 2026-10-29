import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const page = fs.readFileSync(path.join(directory, '..', 'web', 'route_planner.html'), 'utf8');
const setter = page.match(/async function setEndpointFromMap\(prefix,latlng\)\{.*?return true;\}/s)?.[0];

test('map click validates the same rounded coordinates that it stores', async () => {
  assert.ok(setter, 'map point setter exists');

  const fields = { lat: { value: '' }, lon: { value: '' }, id: { value: '' }, name: { value: '' } };
  const calls = [];
  const elements = { accessToken: { value: 'test-token' }, mapPickStatus: { textContent: '' } };
  const context = {
    $: id => elements[id],
    api: async (url, options) => {
      calls.push({ url, payload: JSON.parse(options.body) });
      return {};
    },
    isWithinDemBounds: (lat, lon) => Number.isFinite(lat) && Number.isFinite(lon),
    clearResult() {},
    endpointFields: { start: fields, end: fields },
    renderEndpoints() {},
    showStatus() {},
  };
  const accepted = await vm.runInNewContext(
    `(async()=>{${setter}; return setEndpointFromMap('start',{lat:25.00000049,lng:121.00000051});})()`,
    context,
  );

  assert.equal(accepted, true);
  assert.equal(calls[0].url, '/api/routes/validate-point');
  assert.equal(calls[0].payload.lat, Number(fields.lat.value));
  assert.equal(calls[0].payload.lon, Number(fields.lon.value));
  assert.deepEqual(calls[0].payload, { lat: 25, lon: 121.000001 });
});
