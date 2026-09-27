# Shared-state multi-fact chat check

Five separate `main.py chat` processes used the one shared Titans state from
`checkpoints/session-broad-shared24-full-v1`. They first stored build tools for
two repositories, then corrected one tool. The same 4.2 MB state file was
saved and reloaded between processes. Alpha/Bazel before the correction was
correct; alpha/Ninja after the correction and beta/Meson were both wrong.
The scored result was 1/3. See `results.json` for every prompt and response.

This failure is directly relevant to persistent memory for multiple user or
repository facts. The architecture still needs a reliable way to retain and
select exact updated values.
