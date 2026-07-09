/* Viewer JS for inspect_source's flow map. Expects a global PAYLOAD:
 *   {vps: [{id, lat, lon, n_targets, n_obs}],
 *    tgs: [{id, lat, lon, n_vps, n_obs}],
 *    flows: [{vp, tg, coords: [[lat, lon], [lat, lon]], n_obs, gc_km,
 *             rtt_min?, rtt_med?, weight?}]}
 *
 * Focus model (essence of plot_connectivity_map.py, no recommendations):
 *   click a VP    -> only its flows + its targets stay visible
 *   click a TG    -> only its flows + its VPs stay visible
 *   click a flow  -> only that line + its two endpoints; pair details in panel
 *   Exit button or double-click on the map resets.
 */
(function () {
  const payload = PAYLOAD;

  const map = L.map('map');
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);

  const endpointPts = payload.vps.map((n) => [n.lat, n.lon])
    .concat(payload.tgs.map((n) => [n.lat, n.lon]));
  if (endpointPts.length) {
    map.fitBounds(L.latLngBounds(endpointPts), {padding: [24, 24]});
  } else {
    map.setView([20, 0], 2);
  }

  // Flows first, markers after — later SVG elements render on top, so
  // endpoint clicks win over the (much denser) line mesh.
  const flowLayer = L.layerGroup().addTo(map);
  const vpLayer = L.layerGroup().addTo(map);
  const tgLayer = L.layerGroup().addTo(map);

  const vpById = {};
  const tgById = {};
  const vpMarkers = {};
  const tgMarkers = {};
  const flowLines = [];

  // Dense meshes need faint default lines; sparse sets can afford opaque ones.
  const baseOpacity = Math.min(0.45, Math.max(0.05, 800 / Math.max(payload.flows.length, 1)));

  let selectedVp = null;
  let selectedTg = null;
  let selectedFlow = null; // index into payload.flows
  let panelDiv = null;

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
    return Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  // Markers stay visible in every focus mode; the focused endpoint just
  // gets a darker ring so it stands out from the rest.
  function styleMarker(marker, emphasized) {
    marker.setStyle(emphasized
      ? {weight: 2.5, color: '#111'}
      : {weight: 1, color: marker._baseColor});
    marker.setRadius(marker._baseRadius + (emphasized ? 2 : 0));
  }

  function styleFlow(line, mode) {
    if (mode === 'hidden') {
      line.setStyle({opacity: 0.0, weight: 0.0});
    } else if (mode === 'focus') {
      line.setStyle({color: '#C0392B', opacity: 0.65, weight: 1.8});
    } else if (mode === 'selected') {
      line.setStyle({color: '#7B241C', opacity: 0.95, weight: 4.0});
      if (line.bringToFront) line.bringToFront();
    } else {
      line.setStyle({color: '#C0392B', opacity: baseOpacity, weight: 1.0});
    }
  }

  function updatePanel() {
    if (!panelDiv) return;
    if (selectedVp === null && selectedTg === null && selectedFlow === null) {
      panelDiv.style.display = 'none';
      panelDiv.innerHTML = '';
      return;
    }

    let html =
      '<div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">' +
      '<b>Focus</b>' +
      '<button type="button" data-action="exit-focus" style="padding:3px 8px; border:1px solid #bbb; background:#fff; cursor:pointer;">Exit</button>' +
      '</div>';
    const hint =
      '<div style="color:#555; margin-top:4px;">Click a line for pair details; double-click the map to exit focus.</div>';

    if (selectedFlow !== null) {
      const flow = payload.flows[selectedFlow];
      html +=
        '<div style="margin-top:6px;">' +
        '<b>Flow</b><br>' +
        `<b>VP:</b> ${esc(flow.vp)} (${fmt(flow.coords[0][0], 4)}, ${fmt(flow.coords[0][1], 4)})<br>` +
        `<b>Target:</b> ${esc(flow.tg)} (${fmt(flow.coords[1][0], 4)}, ${fmt(flow.coords[1][1], 4)})<br>` +
        `<b>Observations:</b> ${flow.n_obs}<br>` +
        `<b>Distance:</b> ${fmt(flow.gc_km, 1)} km<br>`;
      if ('rtt_min' in flow) {
        html += `<b>RTT min / median:</b> ${fmt(flow.rtt_min)} / ${fmt(flow.rtt_med)} ms<br>`;
      }
      if ('weight' in flow) {
        html += `<b>Pair weight:</b> ${fmt(flow.weight, 4)}<br>`;
      }
      html += '</div>';
    } else if (selectedVp !== null) {
      const node = vpById[selectedVp];
      html +=
        '<div style="margin-top:6px;">' +
        `<b>VP:</b> ${esc(node.id)}<br>` +
        `<b>Location:</b> ${fmt(node.lat, 4)}, ${fmt(node.lon, 4)}<br>` +
        `<b>Targets:</b> ${node.n_targets}<br>` +
        `<b>Flows:</b> ${node.n_obs}<br>` +
        hint +
        '</div>';
    } else {
      const node = tgById[selectedTg];
      html +=
        '<div style="margin-top:6px;">' +
        `<b>Target:</b> ${esc(node.id)}<br>` +
        `<b>Location:</b> ${fmt(node.lat, 4)}, ${fmt(node.lon, 4)}<br>` +
        `<b>VPs:</b> ${node.n_vps}<br>` +
        `<b>Flows:</b> ${node.n_obs}<br>` +
        hint +
        '</div>';
    }

    panelDiv.style.display = 'block';
    panelDiv.innerHTML = html;
    const exitBtn = panelDiv.querySelector('button[data-action="exit-focus"]');
    if (exitBtn) exitBtn.addEventListener('click', resetFocus);
  }

  function applyVisibility() {
    payload.flows.forEach((flow, i) => {
      let mode;
      if (selectedFlow !== null) mode = (i === selectedFlow) ? 'selected' : 'hidden';
      else if (selectedVp !== null) mode = (flow.vp === selectedVp) ? 'focus' : 'hidden';
      else if (selectedTg !== null) mode = (flow.tg === selectedTg) ? 'focus' : 'hidden';
      else mode = 'default';
      styleFlow(flowLines[i], mode);
    });

    const focusFlow = selectedFlow !== null ? payload.flows[selectedFlow] : null;
    Object.entries(vpMarkers).forEach(([id, marker]) => {
      styleMarker(marker, focusFlow ? id === focusFlow.vp : id === selectedVp);
    });
    Object.entries(tgMarkers).forEach(([id, marker]) => {
      styleMarker(marker, focusFlow ? id === focusFlow.tg : id === selectedTg);
    });

    updatePanel();
  }

  function resetFocus() {
    selectedVp = null;
    selectedTg = null;
    selectedFlow = null;
    applyVisibility();
  }

  payload.flows.forEach((flow, i) => {
    const line = L.polyline(flow.coords, {color: '#C0392B', weight: 1.0, opacity: baseOpacity});
    line.on('click', () => {
      selectedFlow = i;
      selectedVp = null;
      selectedTg = null;
      applyVisibility();
    });
    line.addTo(flowLayer);
    flowLines.push(line);
  });

  payload.vps.forEach((node) => {
    vpById[node.id] = node;
    const marker = L.circleMarker([node.lat, node.lon], {
      radius: 6, color: '#1F77B4', fillColor: '#1F77B4', fillOpacity: 0.9, opacity: 1.0, weight: 1,
    });
    marker._baseColor = '#1F77B4';
    marker._baseRadius = 6;
    marker.on('click', () => {
      selectedVp = node.id;
      selectedTg = null;
      selectedFlow = null;
      applyVisibility();
    });
    marker.addTo(vpLayer);
    vpMarkers[node.id] = marker;
  });

  payload.tgs.forEach((node) => {
    tgById[node.id] = node;
    const marker = L.circleMarker([node.lat, node.lon], {
      radius: 4, color: '#B22222', fillColor: '#E53935', fillOpacity: 0.9, opacity: 1.0, weight: 1,
    });
    marker._baseColor = '#B22222';
    marker._baseRadius = 4;
    marker.on('click', () => {
      selectedTg = node.id;
      selectedVp = null;
      selectedFlow = null;
      applyVisibility();
    });
    marker.addTo(tgLayer);
    tgMarkers[node.id] = marker;
  });

  const statsControl = L.control({position: 'topright'});
  statsControl.onAdd = function () {
    const div = L.DomUtil.create('div', 'leaflet-bar stats-panel');
    const totalObs = payload.flows.reduce((sum, flow) => sum + flow.n_obs, 0);
    div.innerHTML =
      '<b>Dataset</b><br>' +
      `Unique VPs: ${payload.vps.length}<br>` +
      `Unique TGs: ${payload.tgs.length}<br>` +
      `Total flows: ${totalObs}`;
    L.DomEvent.disableClickPropagation(div);
    return div;
  };
  statsControl.addTo(map);

  L.control.layers(null, {
    'VPs': vpLayer,
    'Targets': tgLayer,
    'Flows': flowLayer,
  }, {collapsed: false}).addTo(map);

  const panelControl = L.control({position: 'topright'});
  panelControl.onAdd = function () {
    const div = L.DomUtil.create('div', 'leaflet-bar focus-panel');
    div.style.display = 'none';
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    panelDiv = div;
    return div;
  };
  panelControl.addTo(map);

  map.on('dblclick', resetFocus);
  applyVisibility();
})();
