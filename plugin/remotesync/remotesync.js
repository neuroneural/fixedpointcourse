var RevealRemoteSync = (function () {
  var ws = null;
  var connected = false;
  var room = null;
  var broadcastTimeout = null;

  function isReceiver() {
    var params = new URLSearchParams(window.location.search);
    return params.has('receiver');
  }

  function getRoom() {
    return window.location.pathname.replace(/^\//, '') || 'default';
  }

  function getWsUrl() {
    var loc = window.location;
    var proto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    return proto + '//' + loc.host + '?room=' + encodeURIComponent(room) + '&role=presenter';
  }

  function broadcastState() {
    if (!ws || ws.readyState !== 1) return;
    var state = Reveal.getState();
    var notes = Reveal.getSlideNotes() || null;
    ws.send(JSON.stringify({
      type: 'state',
      state: state,
      notes: notes,
      slideUrl: window.location.pathname
    }));
  }

  function scheduleBroadcast() {
    if (broadcastTimeout) clearTimeout(broadcastTimeout);
    broadcastTimeout = setTimeout(broadcastState, 100);
  }

  function wireHandlers() {
    ws.onopen = function () {
      connected = true;
      console.log('[remotesync] Connected, room=' + room);
      broadcastState();
    };

    ws.onclose = function () {
      connected = false;
      console.log('[remotesync] Disconnected, reconnecting in 3s...');
      setTimeout(initWebSocket, 3000);
    };

    ws.onerror = function () {};

    ws.onmessage = function (event) {
      var msg;
      try { msg = JSON.parse(event.data); } catch { return; }

      if (msg.type === 'navigate') {
        switch (msg.command) {
          case 'next':     Reveal.next();           break;
          case 'prev':     Reveal.prev();           break;
          case 'left':     Reveal.left();           break;
          case 'right':    Reveal.right();          break;
          case 'up':       Reveal.up();             break;
          case 'down':     Reveal.down();           break;
          case 'overview': Reveal.toggleOverview(); break;
          case 'pause':    Reveal.togglePause();    break;
        }
      } else if (msg.type === 'remote-connected') {
        console.log('[remotesync] Remote connected, re-broadcasting');
        broadcastState();
      }
    };
  }

  function initWebSocket() {
    if (isReceiver()) return;
    room = getRoom();
    console.log('[remotesync] Connecting, room=' + room);
    try {
      ws = new WebSocket(getWsUrl());
      wireHandlers();
    } catch (e) {
      setTimeout(initWebSocket, 3000);
    }
  }

  function onChalkboardSend(event) {
    if (!ws || ws.readyState !== 1) return;
    if (event.content && event.content.sender === 'chalkboard-plugin') {
      ws.send(JSON.stringify({ type: 'chalkboard', content: event.content }));
    }
  }

  return {
    id: 'remotesync',

    init: function () {
      if (isReceiver()) return;

      initWebSocket();

      Reveal.addEventListener('slidechanged',   scheduleBroadcast);
      Reveal.addEventListener('fragmentshown',  scheduleBroadcast);
      Reveal.addEventListener('fragmenthidden', scheduleBroadcast);
      Reveal.addEventListener('paused',         scheduleBroadcast);
      Reveal.addEventListener('resumed',        scheduleBroadcast);
      Reveal.addEventListener('overviewshown',  scheduleBroadcast);
      Reveal.addEventListener('overviewhidden', scheduleBroadcast);

      document.addEventListener('send', onChalkboardSend);

      Reveal.addEventListener('ready', function () {
        setTimeout(broadcastState, 500);
      });
    },

    destroy: function () {
      if (ws) { ws.close(); ws = null; }
      connected = false;
      if (broadcastTimeout) clearTimeout(broadcastTimeout);
      Reveal.removeEventListener('slidechanged',   scheduleBroadcast);
      Reveal.removeEventListener('fragmentshown',  scheduleBroadcast);
      Reveal.removeEventListener('fragmenthidden', scheduleBroadcast);
      document.removeEventListener('send', onChalkboardSend);
    }
  };
})();
