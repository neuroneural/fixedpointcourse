const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');
const os = require('os');
const fs = require('fs');
const path = require('path');
const qrcode = require('qrcode-terminal');

const PORT = 3000;
const useTunnel = process.argv.includes('--tunnel');

const app = express();
const server = http.createServer(app);

app.use(express.static(__dirname));

app.get('/diagnostic', (req, res) => {
  const clientIp = req.ip || req.connection.remoteAddress;
  const ips = getLocalIPs();
  res.type('text/html').send(`<!doctype html><html><body style="font-family:monospace;padding:20px">
<h3>Network Diagnostic</h3>
<p>Your IP: <b>${clientIp}</b></p>
<p>Server IPs: ${ips.join(', ')}</p>
<p>Port: ${PORT}</p>
<p>Server time: ${new Date().toISOString()}</p>
<hr>
<p>If you can see this, the server is reachable from your device.</p>
</body></html>`);
});

app.get('/decks', (req, res) => {
  res.json({ decks: getDecks() });
});

const wss = new WebSocketServer({ server });

const rooms = new Map();

wss.on('connection', (ws, req) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const room = url.searchParams.get('room') || 'default';
  const role = url.searchParams.get('role') || 'remote';

  console.log(`  [ws] ${role} connected → room "${room}"`);

  if (!rooms.has(room)) {
    rooms.set(room, { presenters: new Set(), remotes: new Set() });
  }
  const r = rooms.get(room);

  ws._room = room;
  ws._role = role;

  if (role === 'presenter') {
    r.presenters.add(ws);
  } else {
    r.remotes.add(ws);
    for (const p of r.presenters) {
      if (p.readyState === 1) {
        p.send(JSON.stringify({ type: 'remote-connected' }));
      }
    }
  }

  console.log(`  [ws] room "${room}": ${r.presenters.size} presenter(s), ${r.remotes.size} remote(s)`);

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    if (msg.type === 'state') {
      const notes = msg.notes ? ` notes=${msg.notes.length}chars` : ' no-notes';
      console.log(`  [ws] ${ws._role} → state h=${msg.state.indexh} v=${msg.state.indexv}${notes}`);
    } else if (msg.type === 'navigate') {
      console.log(`  [ws] remote → navigate ${msg.command}`);
    } else if (msg.type === 'chalkboard') {
      console.log(`  [ws] presenter → chalkboard ${msg.content.type}`);
    }

    if (ws._role === 'presenter') {
      const payload = JSON.stringify(msg);
      for (const remote of r.remotes) {
        if (remote.readyState === 1) remote.send(payload);
      }
    } else if (ws._role === 'remote') {
      const payload = JSON.stringify(msg);
      for (const presenter of r.presenters) {
        if (presenter.readyState === 1) presenter.send(payload);
      }
    }
  });

  ws.on('close', () => {
    if (ws._role === 'presenter') {
      r.presenters.delete(ws);
    } else {
      r.remotes.delete(ws);
      for (const p of r.presenters) {
        if (p.readyState === 1) {
          p.send(JSON.stringify({ type: 'remote-disconnected' }));
        }
      }
    }
    if (r.presenters.size === 0 && r.remotes.size === 0) {
      rooms.delete(room);
    }
  });
});

function getLocalIPs() {
  const nets = os.networkInterfaces();
  const ips = [];
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        ips.push(net.address);
      }
    }
  }
  return ips;
}

function printQR(url, label) {
  console.log(`    ${label}`);
  console.log();
  qrcode.generate(url, { small: true }, (qr) => {
    console.log(qr);
  });
}

function getDecks() {
  try {
    return fs.readdirSync(__dirname)
      .filter(f => /^lecture\d+\.html$/.test(f))
      .sort((a, b) => {
        const na = parseInt(a.match(/\d+/)[0]);
        const nb = parseInt(b.match(/\d+/)[0]);
        return na - nb;
      });
  } catch {
    return ['lecture1.html'];
  }
}

function printStartup() {
  const ips = getLocalIPs();
  const decks = getDecks();
  const ip = ips[0] || 'localhost';

  console.log(`\n  Slide server running on port ${PORT}\n`);
  console.log(`    Laptop:  http://localhost:${PORT}\n`);

  for (const deck of decks) {
    const label = deck.replace('.html', '').replace('lecture', 'L');
    const phoneUrl = `http://${ip}:${PORT}/speaker-remote.html?deck=${deck}`;
    console.log(`    ${label}:  ${phoneUrl}`);
  }

  console.log();
  printQR(`http://${ip}:${PORT}/speaker-remote.html?deck=${decks[0]}`, `QR for ${decks[0].replace('.html','')}:`);

  if (decks.length > 1) {
    console.log(`    (Change deck= in the URL for other lectures)`);
  }

  console.log();
  console.log(`  ── Phone can't reach the server? ──`);
  console.log(`  Most likely: your WiFi router blocks device-to-device traffic (AP isolation).`);
  console.log();
  console.log(`  Option 1: Create a Mac hotspot (best for presenting):`);
  console.log(`    System Settings → General → Sharing → Internet Sharing`);
  console.log(`    Share from: Wi-Fi  →  To: Wi-Fi`);
  console.log(`    Set a network name + password, then turn it on.`);
  console.log(`    Connect your phone to the new network, then restart this server.`);
  console.log();
  console.log(`  Option 2: Use an internet tunnel (no router changes needed):`);
  console.log(`    node server.js --tunnel`);
  console.log();
  console.log(`  Press Ctrl+C to stop.\n`);
}

async function startTunnel() {
  try {
    require('localtunnel');
  } catch {
    console.error('  localtunnel not installed. Run: npm install localtunnel');
    process.exit(1);
  }

  const localtunnel = require('localtunnel');
  const tunnel = await localtunnel({ port: PORT });
  const decks = getDecks();

  console.log(`\n  Slide server running on port ${PORT}\n`);
  console.log(`    Laptop:  http://localhost:${PORT}\n`);

  for (const deck of decks) {
    const label = deck.replace('.html', '').replace('lecture', 'L');
    const phoneUrl = `${tunnel.url}/speaker-remote.html?deck=${deck}`;
    console.log(`    ${label}:  ${phoneUrl}`);
  }

  console.log();
  printQR(`${tunnel.url}/speaker-remote.html?deck=${decks[0]}`, `QR for ${decks[0].replace('.html','')}:`);

  if (decks.length > 1) {
    console.log(`    (Change deck= in the URL for other lectures)`);
  }

  console.log();
  console.log(`    (Works over the internet — no LAN needed)`);
  console.log();
  console.log(`  Press Ctrl+C to stop.\n`);

  tunnel.on('close', () => {
    console.log('  Tunnel closed.');
  });

  tunnel.on('error', (err) => {
    console.error('  Tunnel error:', err.message);
  });
}

if (useTunnel) {
  server.listen(PORT, '0.0.0.0', startTunnel);
} else {
  server.listen(PORT, '0.0.0.0', printStartup);
}
