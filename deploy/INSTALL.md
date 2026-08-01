# Deploying AI Caller to a new Raspberry Pi

Two separate stages, deliberately kept apart: get it running manually
first, THEN wire up autostart. Don't skip straight to autostart on a
Pi you haven't verified takes a real call correctly — a broken config
that fails silently on every boot is much harder to debug than one
that fails loudly in your terminal.

## Stage 1 — get it running manually

1. Copy this whole project directory onto the new Pi (git clone, or
   copy the folder over) — anywhere you like, e.g.
   `/home/<user>/projects/For Customers`.

2. From inside that directory, run:
   ```bash
   bash deploy/setup_pi.sh
   ```
   This installs PipeWire/WirePlumber/BlueZ system packages, creates a
   `.venv` inside the project with all Python dependencies, and copies
   `local_config.example.py` to `local_config.py` if it doesn't exist
   yet.

3. Follow the three manual steps setup_pi.sh prints at the end:
   fill in `local_config.py`, run the two `gcloud auth` commands, and
   pair/trust the phone(s) that will call this bot via `bluetoothctl`.

4. Test manually — run it in the foreground so you can see everything
   live and kill it with Ctrl+C:
   ```bash
   .venv/bin/python3 ai_caller_final3.py
   ```
   Make a real test call. Confirm the greeting plays, a normal Q&A
   turn works, and hangup is clean. **Don't move to Stage 2 until this
   works.**

## Stage 2 — autostart on boot

Once Stage 1 works, wire it up so a fresh boot needs nothing but
plugging the Pi in and connecting a phone over Bluetooth.

1. Install the service file (as your normal user, not root):
   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/ai-caller.service ~/.config/systemd/user/ai-caller.service
   ```
   **Open `~/.config/systemd/user/ai-caller.service` and check the
   `WorkingDirectory=`/`ExecStart=` paths match where you actually put
   this project** — they default to `~/projects/For Customers`, edit
   them if yours lives somewhere else. This is the file you own and
   tune — nothing here runs until you enable it in the next step.

2. This is a **user** service (see the comment at the top of the unit
   file for why — PipeWire/WirePlumber run per-user here, not
   system-wide). Two commands, one needs `sudo`:
   ```bash
   sudo loginctl enable-linger $(whoami)
   ```
   This is the only privileged step, and it's what makes your user's
   systemd session (and therefore PipeWire/WirePlumber, and therefore
   this service) start at boot even with nobody logged in over
   SSH/console. Without it, the service would only run while you have
   an active session open.

3. Reload and enable (no sudo — this is your user's own systemd
   instance):
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now ai-caller.service
   ```
   `--now` starts it immediately too, so you can verify right away
   without rebooting.

4. Check it's actually running and watch its logs:
   ```bash
   systemctl --user status ai-caller.service
   journalctl --user -u ai-caller -f
   ```
   You should see the same boot sequence as the manual run (client
   warmup, `Ready — waiting for a call`).

5. **Reboot the Pi and confirm it comes up on its own**, with nobody
   logged in — that's the actual test of whether this is real
   plug-and-play:
   ```bash
   sudo reboot
   ```
   Wait ~30-60s, then from another machine: `ssh` in and run
   `systemctl --user status ai-caller.service` — should show
   `active (running)` with no manual intervention.

## Day-to-day commands, once installed

```bash
# Stop it (e.g. to redeploy new code)
systemctl --user stop ai-caller.service

# Start it again
systemctl --user start ai-caller.service

# Restart (stop + start in one step, after deploying new code)
systemctl --user restart ai-caller.service

# Tail live logs
journalctl --user -u ai-caller -f

# Disable autostart entirely (goes back to manual-only)
systemctl --user disable ai-caller.service
```

## Notes / things this deliberately does NOT automate

- **Bluetooth pairing** — needs you physically confirming on the
  phone; scripting this would mean either storing a fixed PIN (most
  phones don't use one for this kind of pairing anyway) or trusting an
  unattended pairing agent, which is a security tradeoff you should
  make consciously per-device, not something baked into a setup
  script.
- **Secrets** (`local_config.py`) — never generated or filled in by
  any script here. You put your own keys in by hand, and the file
  stays gitignored.
- **Multiple phones** — pairing/trusting more than one phone (as
  already done on the dev Pi — 4 phones trusted) just works with this
  setup; the app auto-detects whichever Bluetooth SCO link is active,
  no per-phone config needed.
