# Testing & Verification Plan

Hardware required: 1 Raspberry Pi 4, 1 OpenBCI Cyton board, 1 laptop/desktop (server).

## Day 1: Server-Side Smoke Tests (no hardware needed)

### 1. Dashboard EDF workflow

```bash
streamlit run src/app/physician_app.py
```

- Create a patient profile on the Home page
- Go to New Study, select "EDF File", upload an EDF, run processing
- Confirm Q-score, P-value, MO count charts render on Dashboard
- Verify Export page produces CSV/JSON download

### 2. WebSocket server starts cleanly

```bash
cd src && uvicorn server.ws_server:app --host 0.0.0.0 --port 8765
```

- Confirm `http://localhost:8765/docs` loads (FastAPI auto-docs)
- `curl http://localhost:8765/api/devices` returns `[]`

### 3. Device registration via dashboard

- Open Devices page (`http://localhost:8501/Devices`)
- Register a test device (name: "Pi Lab-01", key: `pi-lab-01`)
- Verify it shows as offline (red indicator)
- Assign a patient to it

## Day 2: Synthetic Pi Integration (no hardware needed)

### 4. Synthetic board end-to-end

Start server:

```bash
cd src && uvicorn server.ws_server:app --host 0.0.0.0 --port 8765
```

Start synthetic Pi (separate terminal):

```bash
cd src && python -m pi.pi_main \
    --synthetic \
    --server ws://localhost:8765/ws/pi/pi-lab-01 \
    --device-id pi-lab-01
```

Verify:
- Server logs show "Device pi-lab-01 connecting", heartbeats arrive every 30s
- On Devices page: status turns green (connected)
- Start a live study from the Devices page or New Study page
- After ~5 minutes, first `feature_update` arrives in server logs
- Dashboard shows Q-score, P-value, MO count data points for the live study

### 5. Config push roundtrip

- Change patient's algorithm window (e.g. 2 min instead of 5 min)
- Push config from Devices page (click "Start Live Study" or use the config push button)
- Verify Pi logs show "Config updated from server" with new window size
- Subsequent processing windows use the new size

### 6. Stop/restart command

- Send "stop" from Devices page
- Verify Pi process exits cleanly, device goes offline on dashboard

## Day 3: Hardware Validation (Pi + Cyton)

### 7. Pi setup

```bash
# On the Pi
pip install -r requirements.txt
python -c "from brainflow.board_shim import BoardShim; print('OK')"
ls /dev/ttyUSB*
```

### 8. Cyton acquisition test (Pi only, no server)

```bash
python src/acquisition/brainflow_stream.py \
    --serial-port /dev/ttyUSB0 --duration 30 --verbose
```

Verify: ~250 samples/sec in logs, no dropped packets over 30 seconds.

### 9. Pi-to-server live streaming

Server on laptop/desktop:

```bash
./scripts/start_server.sh
```

Pi:

```bash
cd src && python -m pi.pi_main \
    --serial-port /dev/ttyUSB0 \
    --server ws://LAPTOP_IP:8765/ws/pi/pi-lab-01 \
    --device-id pi-lab-01
```

- Register device + assign patient + start live study from dashboard
- Verify features arrive every 5 minutes
- Verify Dashboard charts update with new data points

## Day 4: Sustained Run + Edge Cases

### 10. Multi-hour sustained test

- Run Pi streaming for 2+ hours (overnight if possible)
- Verify: no memory leaks on Pi (`htop` — RSS should stay stable)
- Verify: no dropped WebSocket connections in server logs
- Verify: Dashboard hourly summary table populates correctly
- Verify: alerts fire if MO count exceeds the patient's threshold

### 11. Reconnection test

- While streaming: disconnect Pi from network for 30 seconds, then reconnect
- Verify: `ws_client` auto-reconnects (check Pi logs for "reconnecting" message)
- Verify: features resume after reconnection, no gap in data beyond the disconnect window
- Verify: no duplicate feature records in DB (check FeatureRecord timestamps)

### 12. Pi reboot test (requires systemd service)

```bash
sudo cp pi/eeg_streamer.service /etc/systemd/system/
# Edit /etc/systemd/system/eeg_streamer.service — set SERVER_IP and DEVICE_ID
sudo systemctl daemon-reload
sudo systemctl enable eeg_streamer
sudo systemctl start eeg_streamer
```

- `sudo reboot` on Pi
- Verify: service auto-starts after boot, reconnects to server, resumes streaming

## Day 5: Data Quality Validation

### 13. Compare Pi live vs EDF batch results

- Record ~30 minutes of live EEG via Pi streaming
- Simultaneously save raw EEG on the Pi by adding `--log-dir ./logs` to pi_main (or run `brainflow_stream.py` in parallel with `--log-dir`)
- Process the saved raw data through the EDF batch pipeline (New Study > EDF File)
- Compare Q-scores, P-values, and MO counts between live and batch
- Small differences are expected due to timing boundaries, but trends should match

### 14. Multi-patient check (if time permits)

- Register a second device key (e.g. `pi-test-02`), assign to a different patient
- Run both the real Pi and a synthetic Pi concurrently on different device IDs
- Verify data stays separated per patient/study in the DB
- Verify Dashboard shows correct data for each patient/study

## Automated Tests (run daily)

```bash
python -m pytest tests/test_mos.py -v
```

All 10 tests must pass. These cover the MO pipeline end-to-end: bipolar montage, spectrogram, surrogates, band envelopes, GESD, LASSO, q-value, p-value, multi-channel.
